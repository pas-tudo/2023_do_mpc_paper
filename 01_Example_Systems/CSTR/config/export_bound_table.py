# %%
import pandas as pd
import json
# %%

# Read the bounds from the json file
bound_dict = json.load(open('cstr_bounds.json'))
state_df = pd.DataFrame(bound_dict['states']).T
input_df = pd.DataFrame(bound_dict['inputs']).T

combined_df = pd.concat([state_df, input_df], keys=[r'$\vx$', r'$\vu$'], axis=1)
# %%
tex_str = combined_df.to_latex(escape=False)

tex_str =tex_str.replace(r'C_a', r'$c_A$'
    ).replace(r'C_b', r'$c_B$'
    ).replace(r'T_R', r'$T_R$'
    ).replace(r'T_K', r'$T_K$'
    ).replace(r'F', r'$F$'
    ).replace(r'Q_dot', r'$\dot Q$')


tex_list = tex_str.split('\n')
tex_list.pop(1)
tex_list.insert(2, r'\cmidrule(lr){2-5} \cmidrule(lr){6-7}')
tex_list.insert(4,r'{} & \unit{\mol\per\liter} & \unit{\mol\per\liter} & \unit{\celsius} & \unit{\celsius} & \unit{\per\hour} & \unit{\kilo\joule\per\hour}\\')

tex_str = '\n'.join(tex_list)

tex_list
# %%
with open('cstr_bounds_table.tex', 'w') as f:
    f.write(tex_str)

# %%
