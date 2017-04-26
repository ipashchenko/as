import re
import datetime
import pickle
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd


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

    # This marks that \r\n used for separation of blocks
    use_n = False
    try:
        if lines.index('\r\n') == 0:
            lines = lines[1:]
    except ValueError:
        use_n = True
        if lines.index('\n') == 0:
            lines = lines[1:]

    observations = list()
    while True:
        try:
            if not use_n:
                idx = lines.index('\r\n')
            else:
                idx = lines.index('\n')
        except ValueError:
            break
        block = lines[:idx]
        print(block)
        if not block:
            break
        if block[0].startswith('Observational') or\
            block[0].startswith('#') or\
            block[0].startswith('Comments:'):
            observations.append(block)
        lines = lines[idx+1:]
        if not lines:
            break

    return observations


# FIXME: After poor GRT support sometimes goes comment about Lavochkin
def classify_block(block):
    if block[0].startswith('Comments: Lavochkin shortened start time on'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments: Lavochkin shortened stop time on'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments: Lavochkin shrtened stop time on'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments: Lavochkin shortened end time on'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments: Lavochkin shortens the'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments: Lavochkin shortened'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments: shortened for'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('#Comments: Lavochkin cancelled'):
        rank = 1
    elif block[0].startswith('###Comments: Lavochkin cancelled'):
        rank = 1
    elif block[0].startswith('Comments: Lavochkin shifted'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments:Lavochkin shifted'):
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('###Cancelled: 40min too short here'):
        rank = 1
    elif block[0].startswith('Comments: Lavochkin shortened the segment'):
        rank = 1
    elif block[0].startswith('Comments: Lavochkin will give answer later'):
        rank = 1
    elif block[0].startswith('Comments: Lavochkin prolong'):
        rank = 0
    elif block[0].startswith('Comments: Lavochkin extended'):
        rank = 0
    elif block[0].startswith('Observational code:'):
        rank = 0
    elif block[0].startswith('###Cancelled: poor'):
        rank = 0
    elif block[0].startswith('### Cancelled: por'):
        rank = 0
    elif block[0].startswith('###Cancelled by the grav team'):
        rank = 0
    elif block[0].startswith('### Cancelled: poor GRT support'):
        rank = 0
    elif block[0].startswith('### Cancelled: need more GRTs'):
        rank = 0
    elif block[0].startswith('### Cancelled: poor ground support'):
        rank = 0
    elif block[0].startswith('###Cancelled -- poor GRT support'):
        rank = 0
    elif block[0].startswith('###Cancelled since no GBT support'):
        rank = 0
    elif block[0].startswith('###Cancelled: no ground support'):
        rank = 0
    elif block[0].startswith('#Comments: No Ground support'):
        rank = 0
    elif block[0].startswith('###Cancelled: no GRTs'):
        rank = 0
    elif block[0] == '###Cancelled\n':
        rank = 0
    elif block[0].startswith('###Cancelled: PuTS not ready'):
        rank = 0
    elif block[0].startswith('###Cancelled: no GRT support'):
        rank = 0
    elif block[0].startswith('### Cancelled: no ground support'):
        rank = 0
    elif block[0].startswith('### Cancelled: limited GRT support'):
        rank = 0
    elif block[0].startswith('###Cancelled: GBT not available'):
        rank = 0
    elif block[0].startswith('Comments: Observational time changed from'):
        rank = 0
    elif block[0].startswith('###Cancelled: not enough GRT support'):
        rank = 0
    elif block[0].startswith('###Cancelled: not enoiugh GRT support'):
        rank = 0
    elif block[0].startswith('###Cancelled: not tnough GRT support'):
        rank = 0
    elif block[0].startswith('###Cancelled by Lavochkin'):
        rank = 1
    elif block[0].startswith('#Cancelled by Lavochkin'):
        rank = 1
    elif block[0].startswith('### Cancelled by Lavochkin'):
        rank = 1
    elif block[0].startswith('###Cancelld by Lavochkin'):
        rank = 1
    elif ', Lavochkin shortened' in block[0]:
        shortened_time_mins = block[0].strip().split()[-2]
        rank = 1
    elif block[0].startswith('Comments: start time shifted by 30 min -- GRTs: check, please, that it is OK'):
        rank = 0
    else:
        print block
        raise Exception("Check unknown block starting")
    return rank


def get_ut_source_from_block(block):
    for line in block:
        if 'Start(UT)' in line:
            ut_time = "{} {}".format(line.strip().split()[1],
                                     line.strip().split()[2])
            ut_datetime = datetime.datetime.strptime(ut_time, '%d.%m.%Y %H:%M:%S')
        if 'ource:' in line:
            # Take second word because there could be alternative names
            source = line.strip().split()[1]
            # Why they use commas???
            source = source.rstrip(',')

            source_alt = line.strip().split()[-1]
            source_alt = source_alt.rstrip(')')
            source_alt = source_alt.lstrip('(')
    return ut_datetime, source, source_alt


def parse_racat(fname):
    """
    :return: 
        Dictionary with keys - source names and values - instances of
        ``astropy.coordinates.SkyCoord``.
    """

    source_coordinates = dict()

    with open(fname, 'r') as fo:
        lines = fo.readlines()
        for line in lines:
            # if line.strip().startswith('!'):
            #     continue
            try:
                name = re.findall(r".+source='(\S+)'", line)[0]
                ra = re.findall(r".+RA=(\S+) ", line)[0]
                dec = re.findall(r".+DEC=(\S+) ", line)[0]
                source_coordinates.update({str(name):
                                           SkyCoord(ra + ' ' + dec,
                                                    unit=(u.hourangle, u.deg))})
            except IndexError:
                pass

    return source_coordinates


def dump_source_coordinates(ra_cat_fname, outfname):
    """
    Pickle source coordinates to file.
    
    :param ra_cat_fname: 
        Path to RA catalouge.
    :param outfname:
        Path to dump file.
    """
    source_coordinates = parse_racat(ra_cat_fname)
    with open(outfname, 'wb') as fo:
        pickle.dump(source_coordinates, fo)


def load_source_coordinates(fname):
    """
    Load source coordinates dictionary from pickle-file.
    :return: 
        Dictionary with keys - source names and values - instances of
        ``astropy.coordinates.SkyCoord``.
    """
    with open(fname, 'rb') as fo:
        source_coordinates = pickle.load(fo)
    return source_coordinates


def datetime_to_fractional_year(dt):
    return (float(dt.strftime("%j"))-1) / 366


def create_features_responces_dataset(block_sched_files, ra_cat_pkl):
    features_responces = list()
    columns = ('frac_year', 'ra', 'dec', 'rank')
    source_coordinates = load_source_coordinates(ra_cat_pkl)
    for block_sched_file in block_sched_files:
        print("Parsing block-sched file {}".format(block_sched_file))
        blocks = parse_block_schedule_into_blocks(block_sched_file)
        for block in blocks:
            print("Parsing block :")
            print(block)
            rank = classify_block(block)
            try:
                ut_dt, source, source_alt = get_ut_source_from_block(block)
            # FIXME: ??? in coordinates of block schedule
            except ValueError:
                continue
            print source, source_alt
            frac_year = datetime_to_fractional_year(ut_dt)
            try:
                sky_coord = source_coordinates[source]
            except KeyError:
                try:
                    sky_coord = source_coordinates[source_alt]
                # Some weird maser coordinates
                except KeyError:
                    continue
            features_responces.append((frac_year, sky_coord.ra, sky_coord.dec,
                                       rank))
    features_responces = pd.DataFrame.from_records(features_responces,
                                                   columns=columns)
    return features_responces


if __name__ == '__main__':
    import os
    import glob
    block_sched_dir = '/home/ilya/Dropbox/scheduling/block_schedules'
    block_sched_files = glob.glob(os.path.join(block_sched_dir,
                                               '*_block_schedule.*'))
    dump_source_coordinates('/home/ilya/code/as/Radioastron_Input_Catalog_v054.txt',
                            'RA_cat_v054.pkl')
    fr = create_features_responces_dataset(block_sched_files,
                                           'RA_cat_v054.pkl')

    # From astropy docs
    ras = fr[fr['rank'] == 1]['ra']
    decs = fr[fr['rank'] == 1]['dec']
    ras_rad = [ra.wrap_at(180 * u.deg).radian for ra in ras]
    decs_rad = [dec.radian for dec in decs]

    ras_ = fr[fr['rank'] == 0]['ra']
    decs_ = fr[fr['rank'] == 0]['dec']
    ras_rad_ = [ra.wrap_at(180 * u.deg).radian for ra in ras_]
    decs_rad_ = [dec.radian for dec in decs_]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.2))
    plt.subplot(111, projection="aitoff")
    plt.grid(True)
    plt.plot(ras_rad, decs_rad, 'o', markersize=2, alpha=0.2, color='r',
             label='bad')
    plt.plot(ras_rad_, decs_rad_, 'o', markersize=2, alpha=0.2, color='g',
             label='good')
    plt.subplots_adjust(top=0.95, bottom=0.0)
    plt.legend()
    plt.show()

