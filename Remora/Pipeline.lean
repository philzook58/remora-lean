import Remora.Parser
import Remora.Uniquify
import Remora.TypeCheck
import Remora.Intrinsics
import Remora.Interpreter
import Remora.Futhark
import Remora.Monomorphize
import Remora.LambdaLift
import Remora.Defunctionalize
import Remora.Serializer
namespace Remora.Pipeline
private def initial := Uniquify.prepare intrinsicTypes
def uniquify (p : UncheckedProg) : Except String UniqueProg := do
  let (c,_) ← initial
  Uniquify.uniquify c p
def uniquifyExp (e : UncheckedExp) : Except String UniqueExp := do
  let (c,_) ← initial
  Uniquify.uniquifyExp c e
def typeCheck (p : UncheckedProg) : IO (Except String Prog) := do
  match initial with
  | .error e => return .error e
  | .ok (c,ts) => match Uniquify.uniquify c p with
    | .error e => return .error e
    | .ok p => TypeCheck.check ts p
def typeCheckExp (e : UncheckedExp) : IO (Except String Exp) := do
  match initial with
  | .error e => return .error e
  | .ok (c,ts) => match Uniquify.uniquifyExp c e with
    | .error e => return .error e
    | .ok e => TypeCheck.checkExp ts e
private def config : Interpreter.Config := {readExpression := fun input => do
  match Parser.parseExp "<read-file>" input with
  | .error e => return .error e
  | .ok e => typeCheckExp e}
def interpret (entry : String) (args : List Interpreter.Val) (p : UncheckedProg) : IO (Except String Interpreter.Val) := do
  match ← typeCheck p with
  | .error e => return .error e
  | .ok p => Interpreter.interpret entry args p config
def interpretExp (e : UncheckedExp) : IO (Except String Interpreter.Val) := do
  match ← typeCheckExp e with
  | .error e => return .error e
  | .ok e => Interpreter.interpretExp e config
def compile (p : UncheckedProg) : IO (Except String String) := do
  match ← typeCheck p with
  | .error e => return .error e
  | .ok p => return Futhark.compile (Defunctionalize.defunctionalize (LambdaLift.lambdaLift (Monomorphize.monomorphize p)))
def compileExp (e : UncheckedExp) : IO (Except String String) := do
  match ← typeCheckExp e with
  | .error e => return .error e
  | .ok e => return Futhark.compile (LambdaLift.lambdaLiftExp (Monomorphize.monomorphizeExp e))
def lambdaLift (p : UncheckedProg) : IO (Except String Prog) := do
  return (← typeCheck p).map LambdaLift.lambdaLift
def lambdaLiftExp (e : UncheckedExp) : IO (Except String Prog) := do
  return (← typeCheckExp e).map LambdaLift.lambdaLiftExp
def monomorphize (p : UncheckedProg) : IO (Except String Prog) := do
  return (← typeCheck p).map Monomorphize.monomorphize
def monomorphizeExp (e : UncheckedExp) : IO (Except String Exp) := do
  return (← typeCheckExp e).map Monomorphize.monomorphizeExp
def defunctionalize (p : UncheckedProg) : IO (Except String Prog) := do
  return (← typeCheck p).map fun p => Defunctionalize.defunctionalize (LambdaLift.lambdaLift (Monomorphize.monomorphize p))
structure PassAST where
  passName : String
  passJSON : Lean.Json
def compileWithASTs (p : UncheckedProg) : IO (Except String (List PassAST × String)) := do
  let action : ExceptT String IO (List PassAST × String) := do
    let unique ← ExceptT.mk (pure (uniquify p))
    let checked ← ExceptT.mk (typeCheck p)
    let mono := Monomorphize.monomorphize checked
    let lifted := LambdaLift.lambdaLift mono
    let defun := Defunctionalize.defunctionalize lifted
    let ir ← ExceptT.mk (pure (Futhark.compile defun))
    return ([⟨"parse",Lean.toJson p⟩,⟨"uniquify",Lean.toJson unique⟩,
      ⟨"typecheck",Lean.toJson checked⟩,⟨"monomorphize",Lean.toJson mono⟩,
      ⟨"lambdalift",Lean.toJson lifted⟩,⟨"defunctionalize",Lean.toJson defun⟩],ir)
  action.run
def compileWithASTsExp (e : UncheckedExp) : IO (Except String (List PassAST × String)) := do
  let action : ExceptT String IO (List PassAST × String) := do
    let unique ← ExceptT.mk (pure (uniquifyExp e))
    let checked ← ExceptT.mk (typeCheckExp e)
    let mono := Monomorphize.monomorphizeExp checked
    let lifted := LambdaLift.lambdaLiftExp mono
    let defun := Defunctionalize.defunctionalize lifted
    let ir ← ExceptT.mk (pure (Futhark.compile defun))
    return ([⟨"parse",Lean.toJson e⟩,⟨"uniquify",Lean.toJson unique⟩,
      ⟨"typecheck",Lean.toJson checked⟩,⟨"monomorphize",Lean.toJson mono⟩,
      ⟨"lambdalift",Lean.toJson lifted⟩,⟨"defunctionalize",Lean.toJson defun⟩],ir)
  action.run
end Remora.Pipeline
