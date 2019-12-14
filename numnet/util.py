

def convert_number_to_str(number):
    if isinstance(number, int):
        return str(number)

    num_str = '%.3f' % number

    for i in range(3):
        if num_str[-1] == '0':
            num_str = num_str[:-1]
        else:
            break

    if num_str[-1] == '.':
        num_str = num_str[:-1]

    if num_str[0] == '0' and len(num_str) > 1:
        num_str = num_str[1:]

    return num_str
