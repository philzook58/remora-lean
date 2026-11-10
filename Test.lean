import Remora.CLI.Test

def main : IO UInt32 := Remora.CLI.Test.runTests ["tests"]
