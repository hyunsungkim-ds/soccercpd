import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from sympy.combinatorics import Permutation
from sympy.interactive import init_printing
from src.config import *

init_printing(perm_cyclic=True, pretty_print=False)


def decompose_to_cycles(roleperm):
    if roleperm['switch_rate'] > MAX_SWITCH_RATE:
        return []

    perm_list = [0] + [r for (l, r) in sorted([t for t in roleperm[:-1] if str(t) != 'nan'])]
    if len(perm_list) < 11:
        return []

    p = Permutation(perm_list)
    perm_str = str(p)

    result_list = []
    cycles_str = perm_str.split(')')
    for c in cycles_str:
        c = c.replace('(', '')
        result_list.append(c.split(' '))

    return [s for s in result_list if len(s) > 1]


if __name__ == '__main__':
    activity_id = 17985

    fgp_path = f'data/fgp_avg/{activity_id}.csv'
    fgp_df = pd.read_csv(fgp_path, header=0, encoding='utf-8-sig')
    fgp_df['datetime'] = pd.to_datetime(fgp_df['datetime'])
    fgp_df['roleperm'] = fgp_df.apply(lambda x: (x['base_role'], x['role']), axis=1)

    roleperms = fgp_df.pivot_table(
        values='roleperm', index='datetime', columns='player_name', aggfunc='first'
    )
    roleperms['switch_rate'] = fgp_df.groupby('datetime')['switch_rate'].first()

    cycles = roleperms.apply(decompose_to_cycles, axis=1)
    cycle_list = [(dt, ' '.join(c)) for dt, cycles_t in cycles.iteritems() for c in cycles_t]
    cycles_flatten = pd.Series([c for (dt, c) in cycle_list], index=[dt for (dt, c) in cycle_list])

    role_records = pd.read_csv(f'data/role_records.csv', header=0, encoding='utf-8-sig')
    role_record = role_records[role_records['activity_id'] == activity_id]
    role_record['start_dt'] = pd.to_datetime(role_record['start_dt'])
    role_record['end_dt'] = pd.to_datetime(role_record['end_dt'])
    role_time_table = pd.concat([
        role_record.pivot_table(values='session', index='role_period', aggfunc='first'),
        role_record.pivot_table(values='start_dt', index='role_period', aggfunc='min'),
        role_record.pivot_table(values='end_dt', index='role_period', aggfunc='max'),
        role_record.pivot_table(values='formation', index='role_period', aggfunc='first')
        ], axis=1
    )
    role_time_table['session_str'] = role_time_table['session'].apply(lambda x: f"{x}{'st' if x==1 else 'nd'}")
    half_start = role_time_table.groupby('session').apply(lambda x: x['start_dt'].min())
    role_time_table['title'] = role_time_table.apply(
        lambda x: f"{x['session_str']} Half {(x['start_dt'] - half_start[x['session']]).components[2]}'~"
                  f"{(x['end_dt'] - half_start[x['session']]).components[2]}': {'-'.join(x['formation'])}",
        axis=1
    )
    role_position_table = role_record.pivot_table(
        values='aligned_role', index='role_period', columns='base_role', aggfunc='first'
    )

    for i in role_time_table.index:
        role_time = role_time_table.loc[i]
        role_position = role_position_table.loc[i]
        position_cycles = cycles_flatten[(cycles_flatten.index >= role_time['start_dt']) &
                                         (cycles_flatten.index < role_time['end_dt'])].to_frame(name='role')
        position_cycles['position'] = position_cycles['role'].apply(lambda x: '-'.join([role_position[int(s)] for s in x.split(' ')]))
        plt.figure(i)
        vc = position_cycles['position'].value_counts()[:5]
        vc.plot.bar()
        plt.gca().title.set_text(role_time['title'])
        plt.gca().title.set_size(15)
        plt.xticks(rotation=45)
        plt.tight_layout()
        print(position_cycles)


    roleperms
