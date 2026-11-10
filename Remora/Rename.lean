import Remora.Substitute
namespace Remora.Rename
/-- Uniquified names make simultaneous alpha-renaming capture-free. -/
def renameExp (names : VName → VName) (e : Exp) : Exp := Substitute.exp {names} e
def renameBind (names : VName → VName) (b : Bind) : Bind := Substitute.bind {names} b
def renameDecl (names : VName → VName) (d : Decl) : Decl := Substitute.decl {names} d
end Remora.Rename
