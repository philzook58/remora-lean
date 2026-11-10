import Remora.Pipeline
import Remora.Imports
import Remora.Interpreter.Value.Parser
import Regex
import Remora.CLI.Futhark
namespace Remora.CLI.Test
open Remora Remora.Parser Remora.Interpreter
structure Run where
  input : List Parser.SExp := []
  inputFile : Option String := none
  expected : Option (List Parser.SExp) := none
  expectedFile : Option String := none
  expectedError : Option String := none
deriving instance Inhabited for Run
structure Options where
  modes : List String := ["interpret"]
  entries : List String := []
  tags : List String := []
  excludeTags : List String := []
  backend : CLI.Futhark.Backend := .c
  overrideModes : Bool := false
structure Block where
  modes : List String := ["interpret","compile"]
  entry : String := "main"
  tags : List String := []
  runs : List Run := []
private def trim (s : String) := s.trimAscii.toString
private def uncomment (s : String) := trim (String.ofList ((trim s).toList.dropWhile (· == ';')))
private def groups (source : String) : List String := Id.run do
  let mut result := []
  let mut current := []
  for line in source.splitOn "\n" do
    if (trim line).startsWith ";" then current := current ++ [uncomment line]
    else
      if current.contains "==" then result := result ++ [String.intercalate "\n" current]
      current := []
  if current.contains "==" then result := result ++ [String.intercalate "\n" current]
  return result
private partial def directives : List SExp → Block → Except String Block
  | [],b => pure b
  | .atom "tags" _ :: .node '{' xs _ :: rest,b =>
    directives rest {b with tags := xs.map SExp.render}
  | .atom "modes" _ :: .node '{' xs _ :: rest,b => directives rest {b with modes := xs.map SExp.render}
  | .atom "entry:" _ :: .atom name _ :: rest,b => directives rest {b with entry := name}
  | .atom "input" _ :: .node '{' xs _ :: rest,b => directives rest {b with runs := b.runs ++ [{input := xs}]}
  | .atom "input" _ :: .atom "@" _ :: .atom path _ :: rest,b =>
    directives rest {b with runs := b.runs ++ [{inputFile := some path}]}
  | .atom "output" _ :: .node '{' xs _ :: rest,b => do
    let rs := if b.runs.isEmpty then [{ }] else b.runs
    directives rest {b with runs := rs.take (rs.length-1) ++ [{rs.getLast! with expected := some xs}]}
  | .atom "output" _ :: .atom "@" _ :: .atom path _ :: rest,b => do
    let rs := if b.runs.isEmpty then [{ }] else b.runs
    directives rest {b with runs := rs.take (rs.length-1) ++ [{rs.getLast! with expectedFile := some path}]}
  | .atom "error:" _ :: .string pattern _ :: rest,b => do
    let rs := if b.runs.isEmpty then [Run.mk [] none none none none] else b.runs
    directives rest {b with runs := rs.take (rs.length-1) ++ [{rs.getLast! with expectedError := some pattern}]}
  | x::_,_ => throw s!"unrecognized test directive: {x.render}"
def testBlocks (file : System.FilePath) (source : String) : Except String (List Block) :=
  (groups source).mapM fun g => do
    let lines := (g.splitOn "\n").dropWhile (· != "==") |>.drop 1
    let lines := lines.map fun l => if l.startsWith "error:" then
      "error: " ++ (Lean.Json.str ((l.drop 6).toString.trimAscii.toString)).compress else l
    let body := String.intercalate "\n" lines
    let b ← directives (← Parser.tokenize file body) {}
    return if b.runs.isEmpty then {b with runs := [{}]} else b
partial def testFiles (path : System.FilePath) : IO (List System.FilePath) := do
  if ← path.isDir then
    let mut xs := []
    for e in ← path.readDir do xs := xs ++ (← testFiles e.path)
    return xs.mergeSort (fun a b => a.toString ≤ b.toString)
  else return if path.extension == some "remora" then [path] else []
private def values (_file : System.FilePath) (xs : List SExp) : Except String (List Val) :=
  xs.mapM Interpreter.Value.Parser.fromSExp
private def resolveValues (file : System.FilePath) (inline : List SExp) (path : Option String) : IO (Except String (List Val)) := do
  match path with
  | some p => Interpreter.Value.Parser.readValues ((file.parent.getD ".") / p)
  | none => return values file inline

def runTests (paths : List System.FilePath) (options : Options := {}) : IO UInt32 := do
  let mut passed := 0
  let mut failed := 0
  let mut fixtures := 0
  let mut skipped := 0
  let mut count := 0
  for path in paths do
    for file in ← testFiles path do
      count := count + 1
      let src ← IO.FS.readFile file
      match testBlocks file src with
      | .error e => IO.println s!"FAIL {file}: {e}"; failed := failed+1
      | .ok blocks =>
        if blocks.isEmpty then fixtures := fixtures+1
        for block in blocks do
          if !options.entries.isEmpty && !options.entries.contains block.entry then
            skipped := skipped + block.runs.length * options.modes.length
            continue
          if !options.tags.isEmpty && !block.tags.any options.tags.contains then
            skipped := skipped + block.runs.length * options.modes.length
            continue
          if block.tags.any options.excludeTags.contains then
            skipped := skipped + block.runs.length * options.modes.length
            continue
          for mode in options.modes do
           if !options.overrideModes && !block.modes.contains mode then
             skipped := skipped + block.runs.length
             continue
           for run in block.runs do
            let r : ExceptT String IO Unit := do
              let p ← ExceptT.mk (Imports.resolveImports file src)
              let input ← ExceptT.mk (resolveValues file run.input run.inputFile)
              let action : IO (Except String Val) := if mode == "interpret" then Pipeline.interpret block.entry input p else do
                match ← Pipeline.compile p with
                | .error e => return .error e
                | .ok ir => CLI.Futhark.compileAndRun options.backend (file.fileStem.getD "test") ir block.entry (.values input)
              let evaluated ← (monadLift action : ExceptT String IO (Except String Val))
              if let some pattern := run.expectedError then
                let re ← ExceptT.mk (pure (Regex.build (if pattern.isEmpty then ".*" else pattern)))
                match evaluated with
                | .ok v => throw s!"expected an error, got {pretty v}"
                | .error e =>
                  unless (Regex.captures e.toSlice re).isSome do throw s!"error did not match {pattern}: {e}"
                  return ()
              let actual ← ExceptT.mk (pure evaluated)
              if run.expected.isSome || run.expectedFile.isSome then
                let expected ← ExceptT.mk (resolveValues file (run.expected.getD []) run.expectedFile)
                match expected with
                | [v] => unless actual == v do throw s!"expected {pretty v}, got {pretty actual}"
                | _ => throw "expected exactly one output value"
            match ← r.run with
            | .ok _ => passed := passed+1
            | .error e => failed := failed+1; IO.println s!"FAIL {file} [{block.entry}/{mode}]: {e}"
  IO.println s!"{count} files; {passed} runs passed; {failed} failed; {skipped} skipped; {fixtures} files without test blocks"
  return if failed == 0 then 0 else 1
end Remora.CLI.Test
