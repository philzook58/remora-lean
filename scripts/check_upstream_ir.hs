-- Audit-only checker, built against upstream's pinned Futhark sources.
-- This is the same parse/alias-analysis/type-check sequence as `futhark dev`.
import Data.Text.IO qualified as T
import Futhark.Analysis.Alias qualified as Alias
import Futhark.IR.Parse (parseSOACS)
import Futhark.IR.TypeCheck (checkProg)
import System.Environment (getArgs)
import System.Exit (die)

main :: IO ()
main = do
  [file] <- getArgs
  input <- T.readFile file
  case parseSOACS file input of
    Left err -> die (show err)
    Right (_, prog) -> case checkProg (Alias.aliasAnalysis prog) of
      Left err -> die (show err)
      Right () -> pure ()
