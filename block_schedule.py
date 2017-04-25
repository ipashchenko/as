import re
import datetime
import json


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


def get_ut_source_from_block(block):
    for line in block:
        if 'Start(UT)' in line:
            ut_time = "{} {}".format(line.strip().split()[1],
                                     line.strip().split()[2])
            ut_datetime = datetime.strptime(ut_time, '%d.%m.%Y %H:%M:%S')
        if 'Source:' in line:
            source = line.strip().split()[-1]
    return ut_datetime, source


def parse_racat(fname):
    """
    :return: 
        Dictionary with keys - source names and values - tuples of RA & DEC.
    """

    source_coordinates = dict()

    with open(fname, 'r') as fo:
        lines = fo.readlines()
        for line in lines:
            if line.strip().startswith('!'):
                continue
            try:
                name = re.findall(r".+source='(\S+)'", line)[0]
                ra = re.findall(r".+RA=(\S+) ", line)[0]
                dec = re.findall(r".+DEC=(\S+) ", line)[0]
                source_coordinates.update({str(name): (ra, dec)})
            except IndexError:
                pass

    return source_coordinates


def dump_source_coordinates(fname, outfname):
    """
    Dump source coordinates to json format.
    
    :param fname: 
        Path to RA catalouge.
    :param outfname:
        Path to dump file.
    """
    source_coordinates = parse_racat(fname)
    with open(outfname, 'w') as fo:
        json.dump(source_coordinates, fo)


def load_source_coordinates(fname):
    """
    Load source coordinates dictionary from json-file.
    :return: 
        Dictionary with keys - source names and values - lists of RA & DEC.
    """
    with open(fname, 'r') as fo:
        source_coordinates = json.load(fo)
    return source_coordinates

