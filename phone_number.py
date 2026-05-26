#
# phone_number.py
# peter dejonge, with owen pitchford, advised by dr. nate chambers
#
import random

def area_codes(file) -> list:
    pass

def generate_phone(file, N=1) -> list:
    """
    purpose: generate a list with chosen number of random phone numbers, with real area codes
    input: file with area codes
    output: list of lists, containing random phone numbers
    """
    # list for area codes
    area_codes = []
    # will return this list
    phone_numbers = []

    # open the area codes file, and pull out the area codes
    with open(file, 'r') as fh:
        for _ in range(3):
            next(fh)
        
        for line in fh:
            line = line.strip()
            area_codes.append(line)

    # making N random phone numbers. first random area code. then append each number in area
    # code, to the phone number. then append a random int between 2-9 (how phone numbers work)
    # finally, randomly choosing the rest of the numbers in the phone number
    # big help from dr. chambers help on this code
    for _ in range(N):
        number = []
        ac = area_codes[random.randint(0,len(area_codes)-1)]
        for j in range(0,3):
            number.append(int(ac[j])-int('0'))
        # number.append('-')
        number.append(random.randint(2,9))
        for j in range(0,6):
            # if j == 2:
            #     number.append('-')
            number.append(random.randint(0,9))
            j += 1
        nu = ''.join(map(str,number))
        phone_numbers.append(nu)
    return phone_numbers

if __name__ == '__main__':
    numbers = generate_phone('data/area_codes.txt',10)
    print(numbers)
