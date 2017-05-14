import re
import datetime
import pickle
import pprint
from astropy import units as u
from astropy.coordinates import SkyCoord, Latitude, Longitude
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import (PolynomialFeatures, StandardScaler,
                                   LabelEncoder, OneHotEncoder)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from sklearn.feature_extraction import DictVectorizer


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
    columns = ('ut_dt', 'frac_year', 'source_coordinates', 'rank')
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
            frac_year = datetime_to_fractional_year(ut_dt)
            try:
                sky_coord = source_coordinates[source]
            except KeyError:
                try:
                    sky_coord = source_coordinates[source_alt]
                # Some weird maser coordinates
                except KeyError:
                    continue
            features_responces.append((ut_dt, frac_year, sky_coord, rank))
    features_responces = pd.DataFrame.from_records(features_responces,
                                                   columns=columns)
    return features_responces


def parse_orbit(orbit_fname):
    names = ['ut_time', 'x', 'y', 'z', 'xx', 'yy', 'zz']
    df = pd.read_table(orbit_fname, sep='\s+', header=None,
                       names=names, dtype={key: str for key in names},
                       index_col=False)
    for key in names[1:]:
        df[key] = df[key].apply(lambda x: float(x))
    df['dist_km'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    # Filter out seconds equal to 60
    df['sec'] = df['ut_time'].apply(lambda x: x.split(':')[2].split('.')[0])
    bad_ix_array = np.where(df['sec'] == '60')[0]
    df.drop(df.index[bad_ix_array], inplace=True)

    df['ut_time'] =\
        df['ut_time'].apply(lambda x:
                            datetime.datetime.strptime(x.split('.')[0],
                                                       '%Y-%m-%dT%H:%M:%S'))
    del df['xx'], df['yy'], df['zz'], df['sec']
    return df


def parse_orbit_elements(orbit_elements_fname):
    """
    Parse file with orbit elements.
    
    Date/Time UTC, Hpi(1000km), Half(1000km), a(1000km), e, i(degree),
    om(degree), w(degree)
    """
    names = ['ut_time', 'Hpi', 'Half', 'a', 'e', 'i', 'om', 'w']
    df = pd.read_table(orbit_elements_fname, sep='\s+', header=1,
                       names=names, dtype={key: str for key in names},
                       index_col=False)
    for key in names[1:]:
        df[key] = df[key].apply(lambda x: float(x))

    # Filter out seconds equal to 60
    df['sec'] = df['ut_time'].apply(lambda x: x.split(':')[2].split('.')[0])
    bad_ix_array = np.where(df['sec'] == '60')[0]
    df.drop(df.index[bad_ix_array], inplace=True)

    df['ut_time'] = \
        df['ut_time'].apply(lambda x:
                            datetime.datetime.strptime(x.split('.')[0],
                                                       '%Y-%m-%dT%H:%M:%S'))
    return df


def parse_orientations(orient_fname):
    names = ['id', 'obscode', 'ut_time', 'xalpha', 'xdelta', 'yalpha', 'ydelta',
             'zalpha', 'zdelta']
    df = pd.read_table(orient_fname, sep='|', header=2,
                       names=names, dtype={key: str for key in names},
                       index_col=False)
    df['ut_time'] = \
        df['ut_time'].apply(lambda x:
                            datetime.datetime.strptime(x.strip().split('+')[0],
                                                       '%Y-%m-%d %H:%M:%S'))
    for name in names[3:]:
        df[name] = df[name].apply(lambda x: float(x))

    # df['x'] = SkyCoord(df['xalpha'] + ' ' + df['xdelta'],
    #                    unit=(u.hourangle, u.deg))
    for name in ['xalpha', 'yalpha', 'zalpha']:
        df[name] = df[name].apply(lambda x: Longitude(x, u.hourangle))
    for name in ['xdelta', 'ydelta', 'zdelta']:
        df[name] = df[name].apply(lambda x: Latitude(x, u.deg))

    return df


def add_orbit_by_uttime(target_df, orbit_df):
    """
    Function that updates data frame with new orbit-based features.
    
    :param target_df: 
        Data frame with examples containing UT datetime.
    :param orbit_df: 
        Orbit-related data frame created by ``parse_orbit`` function.
    :return: 
        Updated examples data frame.
    """
    distances = list()
    statuses = list()
    x = list()
    y = list()
    z = list()
    row_before = None
    for index, row in target_df.iterrows():
        ut_dt = row['ut_dt']
        ind = (orbit_df.ut_time-ut_dt).abs().argsort()[:1]
        dist_km = orbit_df.ix[ind].dist_km.values[0]
        x.append(orbit_df.ix[ind].x.values[0])
        y.append(orbit_df.ix[ind].y.values[0])
        z.append(orbit_df.ix[ind].z.values[0])
        distances.append(dist_km)
        if row_before is not None:
            ut_dt_before = row_before['ut_dt']
            dist_km_before = orbit_df.ix[(orbit_df.ut_time-ut_dt_before).abs().argsort()[:1]].dist_km.values[0]
        else:
            dist_km_before = dist_km
        if dist_km >= dist_km_before:
            status = 'out'
        else:
            status = 'in'
        row_before = row
        statuses.append(status)

    target_df['dist_km'] = distances
    target_df['status'] = statuses
    target_df['x_km'] = x
    target_df['y_km'] = y
    target_df['z_km'] = z

    return target_df


def add_orbit_elements_by_uttime(target_df, elements_df):
    """
    Function that updates data frame with new orbit elements based features.
    
    :param target_df: 
        Data frame with examples containing UT datetime.
    :param elements_df: 
        Orbit elements related data frame created by ``parse_orbit_elements``
        function.
    :return: 
        Updated examples data frame.
    """
    hpi_ = list()
    half_ = list()
    a_ = list()
    e_ = list()
    i_ = list()
    om_ = list()
    w_ = list()
    for index, row in target_df.iterrows():
        ut_dt = row['ut_dt']
        # First close orbit elements time
        ind_first = (elements_df.ut_time-ut_dt).abs().argsort()[0]
        dt_first = abs(elements_df.ix[ind_first].ut_time - ut_dt)
        # Second close orbit elements time
        ind_second = (elements_df.ut_time-ut_dt).abs().argsort()[1]
        dt_second = abs(elements_df.ix[ind_second].ut_time - ut_dt)
        # Timedelta between first close and second close known elements
        dt_first_second = dt_first + dt_second
        frac_first = dt_first.total_seconds() / dt_first_second.total_seconds()
        frac_second = dt_second.total_seconds() / dt_first_second.total_seconds()
        # Linearly interpolate
        hpi_.append(frac_second * elements_df.ix[ind_first].Hpi +
                    frac_first * elements_df.ix[ind_second].Hpi)
        half_.append(frac_second * elements_df.ix[ind_first].Half +
                     frac_first * elements_df.ix[ind_second].Half)
        a_.append(frac_second * elements_df.ix[ind_first].a +
                  frac_first * elements_df.ix[ind_second].a)
        e_.append(frac_second * elements_df.ix[ind_first].e +
                  frac_first * elements_df.ix[ind_second].e)
        i_.append(frac_second * elements_df.ix[ind_first].i +
                  frac_first * elements_df.ix[ind_second].i)
        om_.append(frac_second * elements_df.ix[ind_first].om +
                  frac_first * elements_df.ix[ind_second].om)
        w_.append(frac_second * elements_df.ix[ind_first].w +
                  frac_first * elements_df.ix[ind_second].w)

    target_df['Hpi'] = hpi_
    target_df['Half'] = half_
    target_df['a'] = a_
    target_df['e'] = e_
    target_df['i'] = i_
    target_df['om'] = om_
    target_df['w'] = w_

    return target_df


def add_orientation_by_uttime(target_df, orient_df):
    """
    Function that updates data frame with new orientation-based features.
    
    :param target_df: 
        Data frame with examples containing UT datetime.
    :param orient_df: 
        Orinetation-related data frame created by ``parse_orientation`` function.
    :return: 
        Updated examples data frame.
    """
    x_orientations = list()
    y_orientations = list()
    z_orientations = list()
    for index, row in target_df.iterrows():
        ix = (orient_df.ut_time-row['ut_dt']).abs().argsort()[:1]

        # Check that time moment is really belongs to experiment
        dt = abs(orient_df.ix[ix].ut_time - row['ut_dt'])
        if (dt > datetime.timedelta(minutes=5)).bool():
            x_orientations.append(None)
            y_orientations.append(None)
            z_orientations.append(None)
            continue

        xalpha = orient_df.ix[ix].xalpha.values[0]
        xdelta = orient_df.ix[ix].xdelta.values[0]
        x_orientations.append(SkyCoord(xalpha, xdelta,
                                       unit=(u.hourangle, u.deg)))
        yalpha = orient_df.ix[ix].yalpha.values[0]
        ydelta = orient_df.ix[ix].ydelta.values[0]
        y_orientations.append(SkyCoord(yalpha, ydelta,
                                       unit=(u.hourangle, u.deg)))
        zalpha = orient_df.ix[ix].zalpha.values[0]
        zdelta = orient_df.ix[ix].zdelta.values[0]
        z_orientations.append(SkyCoord(zalpha, zdelta,
                                       unit=(u.hourangle, u.deg)))

    target_df['x_orientation'] = x_orientations
    target_df['y_orientation'] = y_orientations
    target_df['z_orientation'] = z_orientations

    return target_df


def transform_skycoordinates_to_radec(df, names, move_to_rad=True):
    """
    Transform ``SkyCoordinate`` column to ``sindec``, ``sinra``, ``cosra``.
    
    :param df: 
        Data frame with ``SkyCoordinate`` column.
    :param names:
        Iterable with names of columns with ``SkyCoordinate``s.
    :param move_to_rad: (optional)
        Should be values moved to radians (from degrees)? (default: ``True``)
    :return: 
        Data frame with transformed `SkyCoordinate`` column that has 3 more
        columns now.
    """

    for name in names:
        sindec = list()
        sinra = list()
        cosra = list()
        for index, row in df.iterrows():
            try:
                ra = row[name].ra
                dec = row[name].dec

                if move_to_rad:
                    ra = ra.wrap_at(180 * u.deg).radian
                    dec = dec.radian

                sindec.append(np.sin(dec))
                sinra.append(np.sin(ra))
                cosra.append(np.cos(ra))
            except AttributeError:
                sindec.append(None)
                sinra.append(None)
                cosra.append(None)

        df['sindec_{}'.format(name)] = sindec
        df['sinra_{}'.format(name)] = sinra
        df['cosra_{}'.format(name)] = cosra

    return df


def plot_importance(clf, names):
    """
    Plot features importance of classifier.
    
    :param clf: 
        Instance of classifier with ``feature_importances_`` attribute.
    :param names: 
        Iterable of features names.
    :return: 
        Figure with plot.
    """
    # sort importances
    indices = np.argsort(clf.feature_importances_)
    # plot as bar chart
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(names)), clf.feature_importances_[indices])
    ax.set_yticks(np.arange(len(names)) + 0.25)
    ax.set_yticklabels(np.array(names)[indices])
    ax.set_xlabel('Relative importance')
    fig.show()
    return fig


def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()

    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def one_hot(df, cols):
    """
    The same as in encode_onehot by using ``Pandas`` machinery.
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


# Function used to search best hyperparameters using ``hyperopt``.
def objective(space):
    pprint.pprint(space)
    # clf = LogisticRegression(C=space['C'],
    #                          class_weight={0: 1, 1: space['cw']},
    #                          random_state=1, max_iter=300, n_jobs=1,
    #                          tol=10.**(-5), penalty='l2')
    clf = RandomForestClassifier(n_estimators=space['n_estimators'],
                                 max_depth=space['max_depth'],
                                 max_features=space['max_features'],
                                 min_samples_split=space['mss'],
                                 min_samples_leaf=space['msl'],
                                 class_weight={0: 1, 1: space['cw']},
                                 verbose=1, random_state=1, n_jobs=4)
    # clf = SVC(C=space['C'], class_weight={0: 1, 1: space['cw']},
    #           probability=False, gamma=space['gamma'], random_state=1)
    # clf = KNeighborsClassifier(n_neighbors=space['k'], weights='distance',
    #                            n_jobs=2)
    estimators = list()
    # estimators.append(('poly', PolynomialFeatures()))
    # estimators.append(('scaler', StandardScaler()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    # auc = np.mean(cross_val_score(pipeline, X, y, cv=kfold, scoring='roc_auc',
    #                               verbose=1, n_jobs=1))
    y_preds = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=1)
    CMs = list()
    for train_idx, test_idx in kfold:
        CMs.append(confusion_matrix(y[test_idx], y_preds[test_idx]))
    CM = np.sum(CMs, axis=0)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("TP = {}".format(TP))
    print("FP = {}".format(FP))
    print("FN = {}".format(FN))

    f1 = 2. * TP / (2. * TP + FP + FN)
    # beta = 0.25
    # fbeta = (1+beta**2)*TP/((1+beta**2)*TP+FN*beta**2+FP)

    print("F1: {}".format(f1))
    # print("F0.25: {}".format(fbeta))
    # print("AUC: {}".format(auc))

    return{'loss': 1-f1, 'status': STATUS_OK}


if __name__ == '__main__':
    import os
    import glob
    block_sched_dir = '/home/ilya/Dropbox/scheduling/block_schedules'
    block_sched_files = glob.glob(os.path.join(block_sched_dir,
                                               '*_block_schedule.*'))
    dump_source_coordinates('/home/ilya/Dropbox/AS/Radioastron_Input_Catalog_v054.txt',
                            'RA_cat_v054.pkl')
    orbit_df = parse_orbit('/home/ilya/Dropbox/AS/RA141109-170805.org')
    orient_df = parse_orientations('/home/ilya/Dropbox/AS/orientation_2015_2017.txt')
    elements_df = parse_orbit_elements('/home/ilya/Dropbox/AS/RA150223-171118_orbit_elements.txt')
    fr = create_features_responces_dataset(block_sched_files,
                                           'RA_cat_v054.pkl')
    fr = add_orbit_by_uttime(fr, orbit_df)
    fr = add_orientation_by_uttime(fr, orient_df)
    fr = add_orbit_elements_by_uttime(fr, elements_df)
    names = ['source_coordinates', 'x_orientation', 'y_orientation',
             'z_orientation']
    fr = transform_skycoordinates_to_radec(fr, names)

    # Using distance to SRT and current orbit major axis calculate
    fr['dist_to_Hpi'] = fr['dist_km'].values / (1000. * fr['Hpi'].values)
    fr['dist_to_Half'] = fr['dist_km'].values / (1000. * fr['Half'].values)
    fr['dist_to_a'] = fr['dist_km'].values / (1000. * fr['a'].values)

    # Delete unused columns
    for name in names:
        del fr[name]

    # One-hot encode preceding/receding SRT
    fr = one_hot(fr, ['status'])
    # Delete encoded column
    del fr['status']

    # Create arrays of features and responces
    features_names = ['frac_year', 'dist_km', 'x_km', 'y_km', 'z_km',
                      # Source coordinates
                      'sindec_source_coordinates', 'sinra_source_coordinates',
                      'cosra_source_coordinates',
                      # SRT orientation
                      # 'sindec_x_orientation',
                      # 'sinra_x_orientation',
                      # 'cosra_x_orientation',
                      # 'sindec_y_orientation',
                      # 'sinra_y_orientation',
                      # 'cosra_y_orientation',
                      # 'sindec_z_orientation',
                      # 'sinra_z_orientation',
                      # 'cosra_z_orientation',
                      'status_in', 'status_out',
                      # Orbital elements
                      'Hpi', 'Half', 'a', 'e', 'i', 'om', 'w',
                      'dist_to_Hpi', 'dist_to_Half', 'dist_to_a']

    # Create dataframe with only non-NAN rows of orientations (bad: 57,
    # good: 323) (compare (bad: 297, good: 1008) of full sample.
    # fr = fr[pd.notnull(fr['cosra_z_orientation'])]

    X = np.array(fr[list(features_names)].values, dtype=float)
    y = np.array(fr['rank'].values, dtype=int)

    ##################### Trying classification#################################
    # Just naively try kNN
    # clf = KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=2)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
    #                                                     stratify=y)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))

    # Tuning HP of logistic regression with polynomial features
    # space = {'C': hp.loguniform('C', -3.3, 6.3),
    #          'cw': hp.loguniform('cw', -0.7, 5)}

    # Tuning HP of RF classifier
    space = {'n_estimators': hp.choice('n_estimators', np.arange(100, 1500, 100,
                                                                 dtype=int)),
             'max_depth': hp.choice('max_depth', np.arange(5, 30, dtype=int)),
             'max_features': hp.choice('max_features', np.arange(5, 20, dtype=int)),
             'mss': hp.choice('mss', np.arange(5, 45, 1, dtype=int)),
             'cw': hp.uniform('cw', 1, 30),
             'msl': hp.choice('msl', np.arange(1, 20, dtype=int))}

    # Tuning HP of RBF-SVM
    # space = {'C': hp.loguniform('C', -3.0, 5.6),
    #          'gamma': hp.loguniform('gamma', -6.2, 4.6),
    #          'cw': hp.uniform('cw', 0.5, 20)}

    # Tuning HP of kNN
    # space = {'k': hp.choice('k', np.arange(1, 40, dtype=int))}

    kfold = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=1)

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials)
    pprint.pprint(space_eval(space, best))


    # Plotting two classes #####################################################
    # # From astropy docs
    # ras = fr[fr['rank'] == 1]['ra']
    # decs = fr[fr['rank'] == 1]['dec']
    # ras_rad = [ra.wrap_at(180 * u.deg).radian for ra in ras]
    # decs_rad = [dec.radian for dec in decs]

    # ras_ = fr[fr['rank'] == 0]['ra']
    # decs_ = fr[fr['rank'] == 0]['dec']
    # ras_rad_ = [ra.wrap_at(180 * u.deg).radian for ra in ras_]
    # decs_rad_ = [dec.radian for dec in decs_]

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 4.2))
    # plt.subplot(111, projection="aitoff")
    # plt.grid(True)
    # plt.plot(ras_rad, decs_rad, 'o', markersize=2, alpha=0.2, color='r',
    #          label='bad')
    # plt.plot(ras_rad_, decs_rad_, 'o', markersize=2, alpha=0.2, color='g',
    #          label='good')
    # plt.subplots_adjust(top=0.95, bottom=0.0)
    # plt.legend()
    # plt.show()

    # frac_year = fr[fr['rank'] == 1]['frac_year']
    # frac_year_ = fr[fr['rank'] == 0]['frac_year']
    # plt.figure()
    # plt.subplot(111)
    # plt.hist(frac_year, bins=20, color='r', alpha=0.3, label='bad',
    #          range=[0, 1])
    # plt.hist(frac_year_, bins=20, color='g', alpha=0.3, label='good',
    #          range=[0, 1])
    # plt.xlabel(r'Fraction of the year')
    # plt.ylabel('N')
    # plt.legend()
    # plt.show()
    ############################################################################

