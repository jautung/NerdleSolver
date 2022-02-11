from collections import namedtuple
from enum import Enum, auto
from itertools import groupby
from time import sleep
from typing import List
import itertools
import random

################################################################################
# Constants
################################################################################

NUM_SYMBOLS = 8

################################################################################
# Structs and enums
################################################################################

class Symbol(Enum):
  ZERO = '0'
  ONE = '1'
  TWO = '2'
  THREE = '3'
  FOUR = '4'
  FIVE = '5'
  SIX = '6'
  SEVEN = '7'
  EIGHT = '8'
  NINE = '9'
  PLUS = '+'
  MINUS = '-'
  TIMES = '*'
  DIVIDE = '/'
  EQUAL = '='

class SymbolType(Enum):
  DIGIT = auto()
  OPERATION = auto()

def symbolType(symbol: Symbol) -> SymbolType:
  match symbol:
    case Symbol.ZERO | Symbol.ONE | Symbol.TWO | Symbol.THREE | \
         Symbol.FOUR | Symbol.FIVE | Symbol.SIX | Symbol.SEVEN | \
         Symbol.EIGHT | Symbol.NINE:
      return SymbolType.DIGIT
    case Symbol.PLUS | Symbol.MINUS | Symbol.TIMES | Symbol.DIVIDE | \
         Symbol.EQUAL:
      return SymbolType.OPERATION

equation = namedtuple('equation', ['lhs', 'rhs'])

class ParsingError(Exception):
  pass

################################################################################
# Validity
################################################################################

# Main functions
def isValidGuess(symbols: List[Symbol]) -> bool:
  return isCorrectLength(symbols) and \
         hasSingleEquals(symbols) and \
         isMathematical(symbols, forSolution=False)

def isValidSolution(symbols: List[Symbol]) -> bool:
  return isCorrectLength(symbols) and \
         hasSingleEquals(symbols) and \
         isMathematical(symbols, forSolution=True)

# Helpers
def isCorrectLength(symbols: List[Symbol]) -> bool:
  return len(symbols) == NUM_SYMBOLS

def hasSingleEquals(symbols: List[Symbol]) -> bool:
  return sum(symbol == Symbol.EQUAL for symbol in symbols) == 1

def isMathematical(symbols: List[Symbol], forSolution: bool) -> bool:
  try:
    groupedSymbols = groupSymbols(symbols, forSolution=forSolution)
    if not startsAndEndsWithNumbers(groupedSymbols):
      return False
    equation = parseSymbolsToEquation(groupedSymbols)
    if forSolution and len(equation.rhs) > 1: # atomic rhs for solution
      return False
    lhsValue = evaluate(equation.lhs)
    rhsValue = evaluate(equation.rhs)
    return lhsValue == rhsValue
  except ParsingError:
    return False

# Sub-helpers
def groupSymbols(symbols: List[Symbol], forSolution: bool) -> List:
  groups = [list(sublist) for _, sublist in groupby(symbols, symbolType)]
  return [parseGroup(group, forSolution=forSolution) for group in groups]

def parseGroup(group: List, forSolution: bool) -> any:
  if symbolType(group[0]) == SymbolType.OPERATION:
    if len(group) > 1:
      raise ParsingError
    return group[0]
  elif symbolType(group[0]) == SymbolType.DIGIT:
    if forSolution and group[0] == Symbol.ZERO: # no leading or lone zeroes for solution
      raise ParsingError
    numberString = ''.join([symbol.value for symbol in group])
    try:
      number = int(numberString)
    except ValueError:
      raise ParsingError
    return number

def startsAndEndsWithNumbers(groupedSymbols: List) -> bool:
  return isinstance(groupedSymbols[0], int) and \
         isinstance(groupedSymbols[-1], int)

def parseSymbolsToEquation(groupedSymbols: List) -> equation:
  try:
    equalIndex = groupedSymbols.index(Symbol.EQUAL)
  except ValueError:
    raise ParsingError
  lhs = groupedSymbols[:equalIndex]
  rhs = groupedSymbols[equalIndex+1:]
  return equation(lhs, rhs)

def evaluate(groupedSymbols: List) -> int:
  # This is pretty inefficient but I'm lazy to think of a cleverer method
  currentList = groupedSymbols
  while len(currentList) > 1:
    try:
      indexToEvaluate = next(
        index for index, symbol in enumerate(currentList) if \
        symbol == Symbol.TIMES or symbol == Symbol.DIVIDE
      )
    except StopIteration:
      try:
        indexToEvaluate = next(
          index for index, symbol in enumerate(currentList) if \
          symbol == Symbol.PLUS or symbol == Symbol.MINUS
        )
      except StopIteration:
        indexToEvaluate = None
    currentList = currentList[:indexToEvaluate-1] + \
                  [
                    evaluateTriple(
                      currentList[indexToEvaluate],
                      currentList[indexToEvaluate-1],
                      currentList[indexToEvaluate+1]
                    )
                  ] + \
                  currentList[indexToEvaluate+2:]  
  return currentList[0]

def evaluateTriple(operation: Symbol, lhs: int, rhs: int):
  match operation:
    case Symbol.PLUS:
      return lhs + rhs
    case Symbol.MINUS:
      return lhs - rhs
    case Symbol.TIMES:
      return lhs * rhs
    case Symbol.DIVIDE:
      if rhs == 0:
        raise ParsingError
      return lhs / rhs

################################################################################
# Statistics
################################################################################

################################################################################
# Entropy
################################################################################

################################################################################
# Mains
################################################################################

def generateValidSets():
  validGuessesFile = open('validGuesses.txt', 'w')
  validSolutionsFile = open('validSolutions.txt', 'w')
  for symbols in itertools.product(list(Symbol), repeat=NUM_SYMBOLS):
    validGuess = isValidGuess(symbols)
    validSolution = isValidSolution(symbols)
    if validGuess:
      validGuessesFile.write(' '.join([symbol.value for symbol in symbols]) + '\n')
    if validSolution:
      validSolutionsFile.write(' '.join([symbol.value for symbol in symbols]) + '\n')
  validGuessesFile.close()
  validSolutionsFile.close()

################################################################################
# Tests
################################################################################

def runCraftedValidityTests():
  example0 = [
    Symbol.ONE,
    Symbol.EQUAL,
    Symbol.ONE,
    Symbol.EQUAL,
    Symbol.ONE,
    Symbol.EQUAL,
    Symbol.ZERO,
    Symbol.ONE
  ]
  assert(not isValidGuess(example0))
  assert(not isValidSolution(example0))

  example1 = [
    Symbol.ONE,
    Symbol.PLUS,
    Symbol.ONE,
    Symbol.EQUAL,
    Symbol.ZERO,
    Symbol.ZERO,
    Symbol.ZERO,
    Symbol.TWO
  ]
  assert(isValidGuess(example1))
  assert(not isValidSolution(example1))

  example2 = [
    Symbol.SIX,
    Symbol.DIVIDE,
    Symbol.FOUR,
    Symbol.TIMES,
    Symbol.TWO,
    Symbol.EQUAL,
    Symbol.ZERO,
    Symbol.THREE
  ]
  assert(isValidGuess(example2))
  assert(not isValidSolution(example2))

  example3 = [
    Symbol.NINE,
    Symbol.TIMES,
    Symbol.TWO,
    Symbol.ZERO,
    Symbol.EQUAL,
    Symbol.ONE,
    Symbol.EIGHT,
    Symbol.ZERO
  ]
  assert(isValidGuess(example3))
  assert(isValidSolution(example3))

  example4 = [
    Symbol.ZERO,
    Symbol.PLUS,
    Symbol.FIVE,
    Symbol.PLUS,
    Symbol.FIVE,
    Symbol.EQUAL,
    Symbol.ONE,
    Symbol.ZERO
  ]
  assert(isValidGuess(example4))
  assert(not isValidSolution(example4))

  example5 = [
    Symbol.MINUS,
    Symbol.FIVE,
    Symbol.MINUS,
    Symbol.SIX,
    Symbol.EQUAL,
    Symbol.MINUS,
    Symbol.ONE,
    Symbol.ONE
  ]
  assert(not isValidGuess(example5))
  assert(not isValidSolution(example5))

  example6 = [
    Symbol.EIGHT,
    Symbol.FOUR,
    Symbol.DIVIDE,
    Symbol.SIX,
    Symbol.EQUAL,
    Symbol.NINE,
    Symbol.PLUS,
    Symbol.FIVE
  ]
  assert(isValidGuess(example6))
  assert(not isValidSolution(example6))

  example7 = [
    Symbol.ONE,
    Symbol.ONE,
    Symbol.FOUR,
    Symbol.DIVIDE,
    Symbol.ONE,
    Symbol.NINE,
    Symbol.EQUAL,
    Symbol.SIX
  ]
  assert(isValidGuess(example7))
  assert(isValidSolution(example7))

def runRandomValidityTests():
  while True:
    symbols = [random.choice(list(Symbol)) for _ in range(NUM_SYMBOLS)]
    print(' '.join([symbol.value for symbol in symbols]))
    print('isValidGuess:', isValidGuess(symbols))
    print('isValidSolution:', isValidSolution(symbols))
    if isValidGuess(symbols):
      sleep(3)
    if isValidSolution(symbols):
      break

################################################################################

if __name__ == '__main__':
  # runCraftedValidityTests()
  # runRandomValidityTests()
  generateValidSets()
