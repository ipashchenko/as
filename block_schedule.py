def parse_block_schedule_into_blocks(fname):
    """
    Parse block_schedule into blocks of lines.
    
    :param fname: 
        Path to block_schedule file.
    :return: 
        List of blocks with lines containing information about observation.
    """
    with open(fname, 'r') as fo:
        lines = fo.readlines()

    for line in lines:
        if 'Version' in line:
            break
    version_line_idx = lines.index(line)

    lines = lines[version_line_idx+1:]

    if lines.index('\r\n') == 0:
        lines = lines[1:]

    observations = list()
    while True:
        idx = lines.index('\r\n')
        block = lines[:idx]
        if block[0].startswith('Observational') or\
            block[0].startswith('#') or\
            block[0].startswith('Comments:'):
            observations.append(block)
        lines = lines[idx+1:]
        if not lines:
            break

    return observations


def classify_block(block):
    if block[0].startswith('Comments: Lavochkin shortened start time on'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments: Lavochkin shortened stop time on'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Observational code:'):
        rank = 0
    elif block[0].startswith('###Cancelled: poor'):
        rank = 0
    elif block[0].startswith('###Cancelled by Lavochkin'):
        rank = 1
    else:
        print block
        raise Exception("Check unknown block starting")
    return rank

