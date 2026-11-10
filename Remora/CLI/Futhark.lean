import Remora.Interpreter.Value
import Remora.Interpreter.Value.Parser
namespace Remora.CLI.Futhark
open Remora Remora.Interpreter
inductive Backend where
  | c | cuda
  deriving BEq, Inhabited
inductive Input where
  | values : List Val → Input
  | file : System.FilePath → Input
partial def futharkValue (v : Val) : Except String String := do
  match v with
  | .base (.intVal n) => return s!"{n}i64"
  | .base (.floatVal n) => return s!"{n}f32"
  | .base (.boolVal b) => return if b then "true" else "false"
  | .array [] [v] => futharkValue v
  | .array [0] [] => throw "Futhark input: empty array needs a type annotation"
  | .array [_] vs => return "[" ++ String.intercalate ", " (← vs.mapM futharkValue) ++ "]"
  | .array (d::ds) vs =>
    let rows := Remora.Interpreter.split d.toNat (product ds) vs
    return "[" ++ String.intercalate ", " (← rows.mapM fun row => futharkValue (.array ds row)) ++ "]"
  | _ => throw "Futhark input: unsupported value"
def futharkInput (vs : List Val) : Except String String :=
  return String.intercalate "\n" (← vs.mapM futharkValue) ++ "\n"
def compileBackend (backend : Backend) (dir : System.FilePath) (name ir : String) : IO (Except String String) := do
  try
    IO.FS.createDirAll dir
    let source := dir / (name ++ ".fut_soacs")
    IO.FS.writeFile source ir
    let opts := if backend == .c then #["--seq-mem", "--backend=c"] else #["--gpu-mem", "--backend=cuda"]
    let r ← IO.Process.output {cmd := "futhark", args := #["dev"] ++ opts ++ #[source.toString]}
    return if r.exitCode == 0 then .ok r.stdout else .error (r.stderr ++ r.stdout)
  catch e => return .error s!"futhark: {e}"
def compileAndRun (backend : Backend) (name ir entry : String) (input : Input) : IO (Except String Val) := do
  let dir := System.FilePath.mk s!"/tmp/remora-lean-{← IO.monoMsNow}"
  try
    match ← compileBackend backend dir name ir with
    | .error e => return .error e
    | .ok _ =>
      let vs ← match input with
        | .values vs => pure (.ok vs)
        | .file p => Interpreter.Value.Parser.readValues p
      let .ok vs := vs | return vs.map (fun _ => default)
      let .ok text := futharkInput vs | return .error "unsupported Futhark input"
      let r ← IO.Process.output {cmd := (dir / name).toString, args := #["-e",entry]} (some text)
      if r.exitCode != 0 then return .error (r.stderr ++ r.stdout)
      return Interpreter.Value.Parser.parseVal "<futhark-output>" r.stdout
  catch e => return .error e.toString
  finally
    if ← dir.pathExists then IO.FS.removeDirAll dir
end Remora.CLI.Futhark
