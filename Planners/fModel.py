__author__ = 'rkrsn'
def avoid(name='BDBC'):
  if name == 'Apache':
    return []
  elif name == 'BDBC':
    return [7, 13]
  elif name == 'BDBJ':
    return [0, 1, 2, 5, 6, 10, 13, 14, 16, 17, 18]
  elif name == 'LLVM':
    return [0]
  elif name == 'X264':
    return [0, 8, 12]
  elif name == 'SQL':
    return [0, 2, 7, 10, 23]


def alternates(name='BDBJ'):
  if name == 'Apache':
    return []
  if name == 'BDBC':
    return [range(8, 13), range(14, 18)]
  if name == 'BDBJ':
    return [[11, 12], [3, 4], [7, 8], [23, 24]]
  if name == 'LLVM':
    return []
  if name == 'X264':
    return [[9, 10, 11], [13, 14, 15]]
  if name == 'SQL':
    return [range(3, 7), [25, 27], [28, 29, 30], [32, 33], range(35, 39)]