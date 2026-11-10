import Remora.Pipeline
import Remora.Imports
import Remora.Interpreter.Value.Parser
import Remora.CLI.Futhark
import Remora.CLI.Test
import Remora.Serializer
namespace Remora.CLI
open Remora
structure Options where
  file : Option System.FilePath := none
  expr : Option String := none
  entry : String := "main"
  backend : Option CLI.Futhark.Backend := none
  run : Bool := false
  astsDir : System.FilePath := "."
  flags : List String := []
  args : List String := []
private def parseOptions : List String → Options → Except String Options
  | [], o => pure o
  | "-f"::v::xs,o | "--file"::v::xs,o => parseOptions xs {o with file := some v}
  | "-e"::v::xs,o | "--expr"::v::xs,o => parseOptions xs {o with expr := some v}
  | "--entry"::v::xs,o => parseOptions xs {o with entry := v}
  | "--backend"::v::xs,o => do
    let b ← match v with | "c" => pure CLI.Futhark.Backend.c | "cuda" => pure .cuda | _ => throw "backend must be c or cuda"
    parseOptions xs {o with backend := some b}
  | "--backend=c"::xs,o => parseOptions xs {o with backend := some .c}
  | "--backend=cuda"::xs,o => parseOptions xs {o with backend := some .cuda}
  | "--asts-dir"::v::xs,o => parseOptions xs {o with astsDir := System.FilePath.mk v}
  | "--run"::xs,o => parseOptions xs {o with run := true}
  | x::xs,o => if x.startsWith "--" then
      if ["--asts","--parse","--uniquify","--typecheck","--type-check","--lambdalift","--monomorphize","--defunctionalize","--json"].contains x then
        parseOptions xs {o with flags := o.flags ++ [x]}
      else throw s!"unknown option: {x}"
    else parseOptions xs {o with args := o.args ++ [x]}
private def output [ToString α] : Except String α → IO UInt32
  | .error e => do (← IO.getStderr).putStrLn e; return 1
  | .ok x => do IO.println x; return 0
private def valOutput (r : Except String Interpreter.Val) : IO UInt32 :=
  output (r.map Interpreter.pretty)
private partial def repl : IO UInt32 := do
  let stdout ← IO.getStdout
  stdout.putStr ">> "
  stdout.flush
  let s ← (← IO.getStdin).getLine
  if s.isEmpty then return 0
  match Parser.parseExp "<repl>" s with
  | .error e => IO.println e
  | .ok e => discard (valOutput (← Pipeline.interpretExp e))
  repl
private partial def testOptions : List String → Test.Options → List System.FilePath → Except String (Test.Options × List System.FilePath)
  | [],o,ps => pure (o, if ps.isEmpty then ["tests"] else ps)
  | "--modes"::v::xs,o,ps => testOptions xs {o with modes := v.splitOn ",", overrideModes := true} ps
  | "--entry"::v::xs,o,ps => testOptions xs {o with entries := o.entries ++ v.splitOn ","} ps
  | "--tags"::v::xs,o,ps => testOptions xs {o with tags := o.tags ++ v.splitOn ","} ps
  | "--exclude-tags"::v::xs,o,ps => testOptions xs {o with excludeTags := o.excludeTags ++ v.splitOn ","} ps
  | "--backend=cuda"::xs,o,ps => testOptions xs {o with backend := .cuda} ps
  | "--backend=c"::xs,o,ps => testOptions xs {o with backend := .c} ps
  | x::xs,o,ps => if x.startsWith "--modes=" then testOptions ("--modes"::(x.drop 8).toString::xs) o ps
    else if x.startsWith "-" then throw s!"unknown test option: {x}"
    else testOptions xs o (ps ++ [System.FilePath.mk x])
private def usage := "Remora (Lean)\nUsage: remora repl | parse | interpret | futhark | mlir | dev | test\n  -f FILE, -e EXPR, --entry NAME, --backend=c|cuda, --run\n"
def main (args : List String) : IO UInt32 := do
  let mode := args.headD "--help"
  if mode == "--help" || mode == "-h" || args.contains "--help" then IO.println usage; return 0
  if mode == "repl" then return ← repl
  if mode == "test" then
    match testOptions args.tail! {} [] with
    | .error e => return ← output (.error e : Except String String)
    | .ok (o,paths) => return ← Test.runTests paths o
  let parsed := parseOptions args.tail! {}
  let .ok o := parsed | return ← output (parsed.map (fun _ => ""))
  if mode == "mlir" && (o.run || o.backend.isSome || !o.flags.isEmpty || !o.args.isEmpty) then
    return ← output (.error "mlir supports emission only; execution uses the test harness" : Except String String)
  let (fname,source) ← match o.file,o.expr with
    | some f,none => pure (f,← IO.FS.readFile f)
    | none,some s => pure (System.FilePath.mk "<expression>",s)
    | none,none => pure (System.FilePath.mk "<stdin>",← (← IO.getStdin).readToEnd)
    | _,_ => return ← output (.error "pass either -f or -e" : Except String String)
  if o.expr.isSome then
    match Parser.parseExp fname source with
    | .error e => return ← output (.error e : Except String String)
    | .ok e =>
      if mode == "mlir" then return ← output (← Pipeline.compileMLIRExp e)
      if mode == "interpret" then return ← valOutput (← Pipeline.interpretExp e)
      if mode == "futhark" then
        let compiled ← if o.flags.contains "--asts" then do
          match ← Pipeline.compileWithASTsExp e with
          | .error e => pure (.error e)
          | .ok (asts,ir) =>
            IO.FS.createDirAll o.astsDir
            for ast in asts do IO.FS.writeFile (o.astsDir / (ast.passName ++ ".json")) ast.passJSON.pretty
            pure (.ok ir)
          else Pipeline.compileExp e
        match compiled with
        | .error e => return ← output (.error e : Except String String)
        | .ok ir =>
          if o.run then
            return ← valOutput (← CLI.Futhark.compileAndRun (o.backend.getD .c) "expression" ir o.entry (.values []))
          if let some b := o.backend then return ← output (← CLI.Futhark.compileBackend b "." "expression" ir)
          return ← output (.ok ir)
      if mode == "parse" then IO.println (Lean.toJson e).pretty; return 0
      if mode == "dev" then
        if o.flags.contains "--parse" then return ← output (.ok (Lean.toJson e).pretty)
        if o.flags.contains "--uniquify" then return ← output ((Pipeline.uniquifyExp e).map fun p => (Lean.toJson p).pretty)
        if o.flags.contains "--monomorphize" then return ← output ((← Pipeline.monomorphizeExp e).map fun p => (Lean.toJson p).pretty)
        if o.flags.contains "--lambdalift" then return ← output ((← Pipeline.lambdaLiftExp e).map fun p => (Lean.toJson p).pretty)
        return ← output ((← Pipeline.typeCheckExp e).map fun p => (Lean.toJson p).pretty)
      return ← output (.error s!"unsupported expression mode: {mode}" : Except String String)
  match ← Imports.resolveImports fname source with
  | .error e => return ← output (.error e : Except String String)
  | .ok p =>
    match mode with
    | "parse" => IO.println (Lean.toJson p).pretty; return 0
    | "mlir" => output (← Pipeline.compileMLIR p)
    | "interpret" =>
      match o.args.mapM (Interpreter.Value.Parser.parseVal "<argument>") with
      | .error e => return ← output (.error e : Except String String)
      | .ok vs => return ← valOutput (← Pipeline.interpret o.entry vs p)
    | "futhark" =>
      let compiled ← if o.flags.contains "--asts" then do
        match ← Pipeline.compileWithASTs p with
        | .error e => pure (.error e)
        | .ok (asts,ir) =>
          IO.FS.createDirAll o.astsDir
          for ast in asts do
            let filePrefix := if o.file.isSome then (fname.fileStem.getD "program") ++ "-" else ""
            IO.FS.writeFile (o.astsDir / (filePrefix ++ ast.passName ++ ".json")) ast.passJSON.pretty
          pure (.ok ir)
        else Pipeline.compile p
      match compiled with
      | .error e => return ← output (.error e : Except String String)
      | .ok ir =>
        if o.run then
          let .ok vs := o.args.mapM (Interpreter.Value.Parser.parseVal "<argument>") | return 1
          return ← valOutput (← CLI.Futhark.compileAndRun (o.backend.getD .c) (fname.fileStem.getD "program") ir o.entry (.values vs))
        if let some backend := o.backend then
          let dir := fname.parent.getD "."
          let name := fname.fileStem.getD "remora_program"
          match ← CLI.Futhark.compileBackend backend dir name ir with
          | .error e => return ← output (.error e : Except String String)
          | .ok _ =>
            if o.run then
              let .ok vs := o.args.mapM (Interpreter.Value.Parser.parseVal "<argument>") | return 1
              let .ok input := CLI.Futhark.futharkInput vs | return 1
              let r ← IO.Process.output {cmd := (dir / name).toString, args := #["-e",o.entry]} (some input)
              if r.exitCode != 0 then return ← output (.error r.stderr : Except String String)
              IO.print r.stdout
            return 0
        else return ← output (.ok ir)
    | "dev" =>
      if o.flags.contains "--parse" then return ← output (.ok (Lean.toJson p).pretty)
      if o.flags.contains "--uniquify" then return ← output ((Pipeline.uniquify p).map fun p => (Lean.toJson p).pretty)
      if o.flags.contains "--monomorphize" then return ← output ((← Pipeline.monomorphize p).map fun p => (Lean.toJson p).pretty)
      if o.flags.contains "--lambdalift" then return ← output ((← Pipeline.lambdaLift p).map fun p => (Lean.toJson p).pretty)
      if o.flags.contains "--defunctionalize" then return ← output ((← Pipeline.defunctionalize p).map fun p => (Lean.toJson p).pretty)
      match ← Pipeline.typeCheck p with
      | .error e => return ← output (.error e : Except String String)
      | .ok checked => IO.println (Lean.toJson checked).pretty; return 0
    | _ => return ← output (.error s!"unknown mode: {mode}" : Except String String)
end Remora.CLI

def main (args : List String) : IO UInt32 := do
  try Remora.CLI.main args
  catch e => do (← IO.getStderr).putStrLn (toString e); return 1
