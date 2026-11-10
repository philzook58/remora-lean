import Remora.Parser
namespace Remora.Imports
private partial def resolve (file : System.FilePath) (source : String)
    (seen : IO.Ref (List System.FilePath)) : ExceptT String IO UncheckedProg := do
  let (imports,p) ← ExceptT.mk (pure (Parser.parse file source))
  let mut ds := []
  for i in imports do
    let path ← IO.FS.realPath ((file.parent.getD ".") / i.fileName)
    unless (← seen.get).contains path do
      seen.modify (path :: ·)
      let imported ← resolve path (← IO.FS.readFile path) seen
      ds := ds ++ imported.progDecs
  return ⟨ds ++ p.progDecs⟩
def resolveImports (file : System.FilePath) (source : String) : IO (Except String UncheckedProg) := do
  try
    let seen ← IO.mkRef ([] : List System.FilePath)
    if ← file.pathExists then seen.set [← IO.FS.realPath file]
    (resolve file source seen).run
  catch e => return .error s!"{file}: {e}"
end Remora.Imports
