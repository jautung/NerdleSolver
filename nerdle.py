from collections import namedtuple
from enum import Enum, auto
from itertools import groupby, product
from time import sleep
from typing import List, Tuple, Dict
import random

################################################################################
# Constants
################################################################################

NUM_SYMBOLS = 8
VALID_GUESSES_FILENAME = 'validGuesses.txt'
VALID_SOLUTIONS_FILENAME = 'validSolutions.txt'

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

def characterToSymbol(character: str) -> Symbol:
  match character:
    case '0':
      return Symbol.ZERO
    case '1':
      return Symbol.ONE
    case '2':
      return Symbol.TWO
    case '3':
      return Symbol.THREE
    case '4':
      return Symbol.FOUR
    case '5':
      return Symbol.FIVE
    case '6':
      return Symbol.SIX
    case '7':
      return Symbol.SEVEN
    case '8':
      return Symbol.EIGHT
    case '9':
      return Symbol.NINE
    case '+':
      return Symbol.PLUS
    case '-':
      return Symbol.MINUS
    case '*':
      return Symbol.TIMES
    case '/':
      return Symbol.DIVIDE
    case '=':
      return Symbol.EQUAL

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

class Result(Enum):
  CORRECT = 'C' # green
  MISPLACED = 'M' # yellow
  WRONG = 'W' # gray

equation = namedtuple('equation', ['lhs', 'rhs'])

class ParsingError(Exception):
  pass

################################################################################
# Validity
################################################################################

# Main functions
def generateValidSets():
  validGuessesFile = open(VALID_GUESSES_FILENAME, 'w')
  validSolutionsFile = open(VALID_SOLUTIONS_FILENAME, 'w')
  for symbols in product(list(Symbol), repeat=NUM_SYMBOLS):
    validGuess = isValidGuess(symbols)
    validSolution = isValidSolution(symbols)
    if validGuess:
      validGuessesFile.write(
        ' '.join([symbol.value for symbol in symbols]) + '\n'
      )
    if validSolution:
      validSolutionsFile.write(
        ' '.join([symbol.value for symbol in symbols]) + '\n'
      )
  validGuessesFile.close()
  validSolutionsFile.close()

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
    if forSolution and group[0] == Symbol.ZERO:
      # no leading or lone zeroes for solution
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
# Data I/O
################################################################################

def readDataFile(filename: str) -> List[List[Symbol]]:
  data = []
  dataFile = open(filename, 'r')
  for line in dataFile:
    data.append(parseLine(line))
  dataFile.close()
  return data

def parseLine(line: str) -> List[Symbol]:
  strippedLine = ''.join(line.split())
  return [characterToSymbol(character) for character in strippedLine]

################################################################################
# Statistics
################################################################################

def basicStatistics():
  validSolutions = readDataFile(VALID_SOLUTIONS_FILENAME)
  print(f'Total number of valid solutions: {len(validSolutions)}')

  positionalCounts = [emptySymbolCountMap() for _ in range(NUM_SYMBOLS)]
  for validSolution in validSolutions:
    for index in range(NUM_SYMBOLS):
      positionalCounts[index][validSolution[index]] += 1

  print(f'\nAggregated counts across positions:')
  aggregatedPositionalCount = aggregatePositionalCounts(positionalCounts)
  printPositionalCount(aggregatedPositionalCount)
  for index in range(NUM_SYMBOLS):
    print(f'\nCounts for position {index+1}:')
    printPositionalCount(positionalCounts[index])

  print(f'\nFor csv copying:')
  print(f'Symbol\\Position,', end='')
  print(','.join([str(index+1) for index in range(NUM_SYMBOLS)]))
  for symbol in Symbol:
    print(f'\'{symbol.value},', end='')
    print(','.join([
      str(positionalCounts[index][symbol]) for index in range(NUM_SYMBOLS)
    ]))

def emptySymbolCountMap() -> Dict[Symbol, int]:
  countMap = dict()
  for symbol in Symbol:
    countMap[symbol] = 0
  return countMap

def aggregatePositionalCounts(
  positionalCounts: List[Dict[Symbol, int]]
) -> Dict[Symbol, int]:
  aggregatedPositionalCount = emptySymbolCountMap()
  for positionalCount in positionalCounts:
    for symbol in Symbol:
      aggregatedPositionalCount[symbol] += positionalCount[symbol]
  return aggregatedPositionalCount

def printPositionalCount(positionalCount: Dict[Symbol, int]):
  for symbol in Symbol:
    print(f'> \'{symbol.value}\': {positionalCount[symbol]}')

################################################################################
# Entropy
################################################################################

# Main functions
def runEntropySolving():
  # validGuesses = readDataFile(VALID_GUESSES_FILENAME)
  validSolutions = readDataFile(VALID_SOLUTIONS_FILENAME)
  partitionedSpace = possibleGuessResultsPartitionedSpace(
    validSolutions,
    [
      Symbol.EIGHT,
      Symbol.FOUR,
      Symbol.DIVIDE,
      Symbol.SIX,
      Symbol.EQUAL,
      Symbol.NINE,
      Symbol.PLUS,
      Symbol.FIVE
    ]
  )
  print(len(partitionedSpace))
  # print([
  #   len(subSolutionList)/len(validSolutions) for _, subSolutionList in \
  #   partitionedSpace.items()
  # ])
  for result, subSolutionList in partitionedSpace.items():
    print(' '.join([resultCell.value for resultCell in result]), end='')
    print(f' : {len(subSolutionList)} : ', end='')
    print(' '.join([symbol.value for symbol in subSolutionList[0]]))

# Helpers
def possibleGuessResultsPartitionedSpace(
  solutionList: List[List[Symbol]],
  guess: List[Symbol]
) -> List[float]:
  partitionedSpace = dict()
  for solution in solutionList:
    result = guessResult(solution, guess)
    if result in partitionedSpace:
      partitionedSpace[result].append(solution)
    else:
      partitionedSpace[result] = [solution]
  return partitionedSpace

# Sub-helpers
def guessResult(solution: List[Symbol], guess: List[Symbol]) -> Tuple[Result]:
  # Again, not the most efficient, but I'm lazy to think of a cleverer method
  result = [None for _ in range(NUM_SYMBOLS)]
  solutionTemp = solution.copy()
  for index in range(NUM_SYMBOLS):
    if guess[index] == solution[index]:
      result[index] = Result.CORRECT
      solutionTemp.remove(solution[index])
    elif guess[index] not in solution:
      result[index] = Result.WRONG
  for index in range(NUM_SYMBOLS):
    if result[index] == None:
      if guess[index] in solutionTemp:
        result[index] = Result.MISPLACED
        solutionTemp.remove(guess[index])
      else:
        result[index] = Result.WRONG
  return tuple(result)

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
    print(f'isValidGuess: {isValidGuess(symbols)}')
    print(f'isValidSolution: {isValidSolution(symbols)}')
    print()
    if isValidGuess(symbols):
      sleep(3)
    if isValidSolution(symbols):
      break

def runCraftedGuessResultTests():
  assert(isMatchGuessResult(
    [
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO
    ],
    [
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO
    ],
    [
      Result.CORRECT,
      Result.CORRECT,
      Result.CORRECT,
      Result.CORRECT,
      Result.CORRECT,
      Result.CORRECT,
      Result.CORRECT,
      Result.CORRECT
    ]
  ))

  assert(isMatchGuessResult(
    [
      Symbol.ONE,
      Symbol.ONE,
      Symbol.ONE,
      Symbol.ONE,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO
    ],
    [
      Symbol.ONE,
      Symbol.ONE,
      Symbol.ZERO,
      Symbol.TWO,
      Symbol.ZERO,
      Symbol.TWO,
      Symbol.ONE,
      Symbol.ONE
    ],
    [
      Result.CORRECT,
      Result.CORRECT,
      Result.MISPLACED,
      Result.WRONG,
      Result.CORRECT,
      Result.WRONG,
      Result.MISPLACED,
      Result.MISPLACED
    ]
  ))

  assert(isMatchGuessResult(
    [
      Symbol.ONE,
      Symbol.ONE,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO,
      Symbol.ZERO
    ],
    [
      Symbol.ONE,
      Symbol.TWO,
      Symbol.TWO,
      Symbol.TWO,
      Symbol.TWO,
      Symbol.TWO,
      Symbol.ONE,
      Symbol.ONE
    ],
    [
      Result.CORRECT,
      Result.WRONG,
      Result.WRONG,
      Result.WRONG,
      Result.WRONG,
      Result.WRONG,
      Result.MISPLACED,
      Result.WRONG
    ]
  ))

def runRandomGuessResultTests():
  for _ in range(5):
    solution = [random.choice(list(Symbol)) for _ in range(NUM_SYMBOLS)]
    guess = [random.choice(list(Symbol)) for _ in range(NUM_SYMBOLS)]
    print('Solution:', ' '.join([symbol.value for symbol in solution]))
    print('Guess   :', ' '.join([symbol.value for symbol in guess]))
    result = guessResult(solution, guess)
    print('Result  :', ' '.join([resultCell.value for resultCell in result]))
    print()

def isMatchGuessResult(
  solution: List[Symbol],
  guess: List[Symbol],
  result: Tuple[Result]
) -> bool:
  realResult = guessResult(solution, guess)
  for index in range(NUM_SYMBOLS):
    if realResult[index] != result[index]:
      return False
  return True

################################################################################

if __name__ == '__main__':
  # runCraftedValidityTests()
  # runRandomValidityTests()
  # generateValidSets()
  # basicStatistics()
  # runCraftedGuessResultTests()
  # runRandomGuessResultTests()
  runEntropySolving()
