# Enables inline-plot rendering
# Utilized to create and work with dataframes
import sys
import pandas as pd
import numpy as np
# MATPLOTLIB
import matplotlib.pyplot as plt
import MySQLdb


def db_to_df():
    # Must have Apache, mySQL admin online and running
    conn = MySQLdb.connect(host="localhost",
                           user="root",
                           passwd="",
                           db="freezecasting_opendata")
    # Create proxy cursor to interact with the database
    cursor = conn.cursor()

    # IMPORT AUTHORS TABLE
    authors = pd.read_sql('SELECT * FROM authors', conn)
    authors.columns = ['author_ID', 'name_author', 'affiliation', 'city', 'country', 'long', 'lat', 'email', 'last_updated']

    # drop last_update column
    authors = authors.drop('last_updated', axis=1)
    # Author_ID is imported as float; change to integer data type
    authors['author_ID'] = authors['author_ID'].astype('int')

    # IMPORT PAPERS TABLE
    papers = pd.read_sql('SELECT * FROM papers', conn)
    papers.columns = ['paper_ID', 'author_ID', 'authors', 'title', 'journal', 'volume', 'issue',
                      'pages', 'year', 'doi', 'material', 'material_group', 'last_updated']
    # column names; i.e., variable names, can be changed below (just be sure to maintain order;
    # see pandas documentation for reordering columns).

    # Drop last_updated column
    # --A last_updated column is contained within all tables; it is an automatic timestamp that
    # shows the last time the corresponding row was updated. These columns are unnecessary here,
    # so we will drop the column from each table.
    papers = papers.drop('last_updated', axis=1)
    papers['year'] = papers['year'].astype('int')
    papers['paper_ID'] = papers['paper_ID'].astype('int')

    # IMPORT SAMPLES TABLE
    samples = pd.read_sql('SELECT * FROM samples', conn)
    samples.columns = ['sample_ID', 'paper_ID', 'material_ID', 'vf_total', 'particles_total', 'fluids_total', 'composite',
                       'last_updated']
    # drop last_update column
    samples = samples.drop('last_updated', axis=1)
    samples['sample_ID'] = samples['sample_ID'].astype('int')

    # Merge papers and authors tables to link corresponding author contact data
    # --author_ID acts as primary key for authors table and foreign key for papers table
    papers = pd.merge(papers, authors, on='author_ID', how='outer')

    # Merge papers and samples table to link sample data to citation data
    # --paper_ID acts as primary key for papers table and foreign key for samples table
    samples = pd.merge(samples, papers, on='paper_ID', how='outer')

    # Uncomment and run the following to verify column names and/or print the dataframe header
    # samples.columns.values
    # samples.head()

    data = samples[(samples['sample_ID']) > 0]

    # IMPORT SUSPENSION TABLE
    suspension = pd.read_sql('SELECT * FROM suspension', conn)
    suspension.columns = ['suspension_ID', 'sample_ID', 'time_mill', 'pH_susp', 'viscosity_susp', 'zeta_susp',
                          'last_updated']

    # drop last_update column
    suspension = suspension.drop('last_updated', axis=1)
    # suspension_ID is a primary key for the suspension table; it is not needed here
    suspension = suspension.drop('suspension_ID', axis=1)

    # -----FLUID TABLES-----#
    # Fluid properties table
    props_fluids = pd.read_sql('SELECT * FROM props_fluids', conn)
    props_fluids.columns = ['props_fluid_ID', 'name_fluid', 'density_liq_fluid', 'density_sol_fluid',
                            'thermal_cond_liq_fluid',
                            'thermal_cond_sol_fluid', 'last_updated']
    props_fluids = props_fluids.drop('last_updated', axis=1)
    # Fluid 1
    susp_fluid_1 = pd.read_sql('SELECT * FROM susp_fluid_1', conn)
    susp_fluid_1.columns = ['fluid_1_ID', 'sample_ID', 'props_fluid_ID', 'vf_fluid_1', 'last_updated']
    susp_fluid_1 = susp_fluid_1.drop('last_updated', axis=1)
    susp_fluid_1 = susp_fluid_1.drop('fluid_1_ID', axis=1)
    # Merge fluid properties table with fluid 1 table; 'props_fluid_ID' is primary key
    # for the fluid properties table and foreign key for the susp_fluid_1 table.
    susp_fluid_1 = pd.merge(props_fluids, susp_fluid_1, on='props_fluid_ID', how='outer')
    # rename the 'props_fluid_id' and other property columns such that they can be differentiated for fluid 1 and fluid 2
    susp_fluid_1.rename(columns={'props_fluid_ID': 'props_fluid_ID1'}, inplace=True)
    susp_fluid_1.rename(columns={'name_fluid': 'name_fluid1'}, inplace=True)
    susp_fluid_1.rename(columns={'density_liq_fluid': 'density_liq_fluid1'}, inplace=True)
    susp_fluid_1.rename(columns={'density_sol_fluid': 'density_sol_fluid1'}, inplace=True)
    susp_fluid_1.rename(columns={'thermal_cond_liq_fluid': 'thermal_cond_liq_fluid1'}, inplace=True)
    susp_fluid_1.rename(columns={'thermal_cond_sol_fluid': 'thermal_cond_sol_fluid1'}, inplace=True)

    # Fluid 2

    susp_fluid_2 = pd.read_sql('SELECT * FROM susp_fluid_2', conn)
    susp_fluid_2.columns = ['fluid_2_ID', 'sample_ID', 'props_fluid_ID',
                            'vf_fluid_2', 'last_updated']
    susp_fluid_2 = susp_fluid_2.drop('last_updated', axis=1)
    susp_fluid_2 = susp_fluid_2.drop('fluid_2_ID', axis=1)
    susp_fluid_2 = pd.merge(props_fluids, susp_fluid_2, on='props_fluid_ID', how='outer')
    susp_fluid_2.rename(columns={'props_fluid_ID': 'props_fluid_ID2'}, inplace=True)
    susp_fluid_2.rename(columns={'name_fluid': 'name_fluid2'}, inplace=True)
    susp_fluid_2.rename(columns={'density_liq_fluid': 'density_liq_fluid2'}, inplace=True)
    susp_fluid_2.rename(columns={'density_sol_fluid': 'density_sol_fluid2'}, inplace=True)
    susp_fluid_2.rename(columns={'thermal_cond_liq_fluid': 'thermal_cond_liq_fluid2'}, inplace=True)
    susp_fluid_2.rename(columns={'thermal_cond_sol_fluid': 'thermal_cond_sol_fluid2'}, inplace=True)

    # Merge fluids tables
    fluids = pd.merge(susp_fluid_1, susp_fluid_2, on='sample_ID', how='outer')
    fluids = fluids[(fluids['sample_ID']) > 0]

    # Merge suspension table
    suspension = pd.merge(fluids, suspension, on='sample_ID', how='outer')

    # -----PARTICLE TABLES-----#
    # Particle properties table

    props_particles = pd.read_sql('SELECT * FROM props_particles', conn)
    props_particles.columns = ['props_part_ID', 'name_part', 'description_part', 'density_part', 'thermal_cond_part',
                               'last_updated']
    props_particles = props_particles.drop('last_updated', axis=1)
    # Particle 1
    susp_part_1 = pd.read_sql('SELECT * FROM susp_part_1', conn)
    susp_part_1.columns = ['particle_1_ID', 'sample_ID', 'props_part_ID', 'shape_part_1', 'dia_part_1', 'length_part_1',
                           'vf_part_1', 'last_updated']
    susp_part_1 = susp_part_1.drop('last_updated', axis=1)
    susp_part_1 = susp_part_1.drop('particle_1_ID', axis=1)
    # Merge particle properties table with particle 1 table; 'props_part_1' is primary key
    # for the particle properties table and foreign key for the susp_part_1 table.
    susp_part_1 = pd.merge(props_particles, susp_part_1, on='props_part_ID', how='outer')
    # rename the 'props_part_id' column and other property columns such that it can be differentiated for particle 1 and particle 2
    susp_part_1.rename(columns={'props_part_ID': 'props_part_ID1'}, inplace=True)
    susp_part_1.rename(columns={'name_part': 'name_part1'}, inplace=True)
    susp_part_1.rename(columns={'description_part': 'description_part1'}, inplace=True)
    susp_part_1.rename(columns={'density_part': 'density_part1'}, inplace=True)
    susp_part_1.rename(columns={'thermal_cond_part': 'thermal_cond_part1'}, inplace=True)
    susp_part_1 = susp_part_1[(susp_part_1['sample_ID']) > 0]

    # Particle 2
    susp_part_2 = pd.read_sql('SELECT * FROM susp_part_2', conn)
    susp_part_2.columns = ['particle_2_ID', 'sample_ID', 'props_part_ID', 'shape_part_2', 'dia_part_2', 'length_part_2',
                           'vf_part_2', 'last_updated']
    susp_part_2 = susp_part_2.drop('last_updated', axis=1)
    susp_part_2 = susp_part_2.drop('particle_2_ID', axis=1)
    # Merge particle properties table with particle 2 table; 'props_part_2' is primary key
    # for the particle properties table and foreign key for the susp_part_2 table.
    susp_part_2 = pd.merge(props_particles, susp_part_2, on='props_part_ID', how='outer')
    # rename the 'props_part_id' column and other property columns such that it can be differentiated for particle 1 and particle 2
    susp_part_2.rename(columns={'props_part_ID': 'props_part_ID2'}, inplace=True)
    susp_part_2.rename(columns={'name_part': 'name_part2'}, inplace=True)
    susp_part_2.rename(columns={'description_part': 'description_part2'}, inplace=True)
    susp_part_2.rename(columns={'density_part': 'density_part2'}, inplace=True)
    susp_part_2.rename(columns={'thermal_cond_part': 'thermal_cond_part2'}, inplace=True)
    susp_part_2 = susp_part_2[(susp_part_2['sample_ID']) > 0]

    # Merge particle tables, then merge particle table with suspension table
    particles = pd.merge(susp_part_1, susp_part_2, on='sample_ID', how='outer')
    suspension = pd.merge(particles, suspension, on='sample_ID', how='outer')

    # -----ADDITIVE TABLES-----#
    # Additive properties table
    props_adds = pd.read_sql('SELECT * FROM props_adds', conn)
    props_adds.columns = ['props_add_ID', 'name_add', 'density_add', 'molecular_wt_add']

    # Binder 1
    susp_bind_1 = pd.read_sql('SELECT * FROM susp_bind_1', conn)
    susp_bind_1.columns = ['bind_1_ID', 'sample_ID', 'props_add_ID', 'wf_bind_1', 'last_updated']
    susp_bind_1 = susp_bind_1.drop('last_updated', axis=1)
    susp_bind_1 = susp_bind_1.drop('bind_1_ID', axis=1)
    susp_bind_1 = pd.merge(props_adds, susp_bind_1, on='props_add_ID', how='outer')
    susp_bind_1.rename(columns={'props_add_ID': 'props_bind1_ID'}, inplace=True)
    susp_bind_1.rename(columns={'name_add': 'name_bind1'}, inplace=True)
    susp_bind_1.rename(columns={'density_add': 'density_bind1'}, inplace=True)
    susp_bind_1.rename(columns={'molecular_wt_add': 'molecular_wt_bind1'}, inplace=True)
    susp_bind_1 = susp_bind_1[(susp_bind_1['sample_ID']) > 0]
    # Binder 2
    susp_bind_2 = pd.read_sql('SELECT * FROM susp_bind_2', conn)
    susp_bind_2.columns = ['bind_2_ID', 'sample_ID', 'props_add_ID', 'wf_bind_2', 'last_updated']
    susp_bind_2 = susp_bind_2.drop('last_updated', axis=1)
    susp_bind_2 = susp_bind_2.drop('bind_2_ID', axis=1)
    susp_bind_2 = pd.merge(props_adds, susp_bind_2, on='props_add_ID', how='outer')
    susp_bind_2.rename(columns={'props_add_ID': 'props_bind2_ID'}, inplace=True)
    susp_bind_2.rename(columns={'name_add': 'name_bind2'}, inplace=True)
    susp_bind_2.rename(columns={'density_add': 'density_bind2'}, inplace=True)
    susp_bind_2.rename(columns={'molecular_wt_add': 'molecular_wt_bind2'}, inplace=True)
    susp_bind_2 = susp_bind_2[(susp_bind_2['sample_ID']) > 0]
    # Dispersant 1
    susp_disp_1 = pd.read_sql('SELECT * FROM susp_disp_1', conn)
    susp_disp_1.columns = ['disp_1_ID', 'sample_ID', 'props_add_ID', 'wf_disp_1', 'last_updated']
    susp_disp_1 = susp_disp_1.drop('last_updated', axis=1)
    susp_disp_1 = susp_disp_1.drop('disp_1_ID', axis=1)
    susp_disp_1 = pd.merge(props_adds, susp_disp_1, on='props_add_ID', how='outer')
    susp_disp_1.rename(columns={'props_add_ID': 'props_disp1_ID'}, inplace=True)
    susp_disp_1.rename(columns={'name_add': 'name_disp_1'}, inplace=True)
    susp_disp_1.rename(columns={'density_add': 'density_disp_1'}, inplace=True)
    susp_disp_1.rename(columns={'molecular_wt_add': 'molecular_wt_disp_1'}, inplace=True)
    susp_disp_1 = susp_disp_1[(susp_disp_1['sample_ID']) > 0]
    # Dispersant 2
    susp_disp_2 = pd.read_sql('SELECT * FROM susp_disp_2', conn)
    susp_disp_2.columns = ['disp_2_ID', 'sample_ID', 'props_add_ID', 'wf_disp_2', 'last_updated']
    susp_disp_2 = susp_disp_2.drop('last_updated', axis=1)
    susp_disp_2 = susp_disp_2.drop('disp_2_ID', axis=1)
    susp_disp_2 = pd.merge(props_adds, susp_disp_2, on='props_add_ID', how='outer')
    susp_disp_2.rename(columns={'props_add_ID': 'props_disp2_ID'}, inplace=True)
    susp_disp_2.rename(columns={'name_add': 'name_disp2'}, inplace=True)
    susp_disp_2.rename(columns={'density_add': 'density_disp_2'}, inplace=True)
    susp_disp_2.rename(columns={'molecular_wt_add': 'molecular_wt_disp_2'}, inplace=True)
    susp_disp_2 = susp_disp_2[(susp_disp_2['sample_ID']) > 0]
    # Cryoprotectant
    susp_cryo = pd.read_sql('SELECT * FROM susp_cryo', conn)
    susp_cryo.columns = ['cryo_ID', 'sample_ID', 'props_add_ID', 'wf_cryo', 'last_updated']
    susp_cryo = susp_cryo.drop('last_updated', axis=1)
    susp_cryo = susp_cryo.drop('cryo_ID', axis=1)
    susp_cryo = pd.merge(props_adds, susp_cryo, on='props_add_ID', how='outer')
    susp_cryo.rename(columns={'props_add_ID': 'props_cryo_ID'}, inplace=True)
    susp_cryo.rename(columns={'name_add': 'name_cryo'}, inplace=True)
    susp_cryo.rename(columns={'density_add': 'density_cryo'}, inplace=True)
    susp_cryo.rename(columns={'molecular_wt_add': 'molecular_wt_cryo'}, inplace=True)
    susp_cryo = susp_cryo[(susp_cryo['sample_ID']) > 0]
    # Surfactant
    susp_surfact = pd.read_sql('SELECT * FROM susp_surfact', conn)
    susp_surfact.columns = ['surfact_ID', 'sample_ID', 'props_add_ID', 'wf_surfact', 'last_updated']
    susp_surfact = susp_surfact.drop('last_updated', axis=1)
    susp_surfact = susp_surfact.drop('surfact_ID', axis=1)
    susp_surfact = pd.merge(props_adds, susp_surfact, on='props_add_ID', how='outer')
    susp_surfact.rename(columns={'props_add_ID': 'props_surfact_ID'}, inplace=True)
    susp_surfact.rename(columns={'name_add': 'name_surfact'}, inplace=True)
    susp_surfact.rename(columns={'density_add': 'density_surfact'}, inplace=True)
    susp_surfact.rename(columns={'molecular_wt_add': 'molecular_wt_surfact'}, inplace=True)
    susp_surfact = susp_surfact[(susp_surfact['sample_ID']) > 0]

    # Merge suspension tables
    suspension = pd.merge(suspension, susp_bind_1, on='sample_ID', how='outer')
    suspension = pd.merge(suspension, susp_bind_2, on='sample_ID', how='outer')
    suspension = pd.merge(suspension, susp_disp_1, on='sample_ID', how='outer')
    suspension = pd.merge(suspension, susp_disp_2, on='sample_ID', how='outer')
    suspension = pd.merge(suspension, susp_cryo, on='sample_ID', how='outer')
    suspension = pd.merge(suspension, susp_surfact, on='sample_ID', how='outer')

    # Uncomment and run the code block below if freeze gel-casting additive tables were imported
    # suspension = pd.merge(suspension, susp_catal_1, on = 'sample_ID', how = 'outer')
    # suspension = pd.merge(suspension, susp_cross_1, on = 'sample_ID', how = 'outer')
    # suspension = pd.merge(suspension, susp_init_2, on = 'sample_ID', how = 'outer')
    # suspension = pd.merge(suspension, susp_mono, on = 'sample_ID', how = 'outer')

    # Merge suspension table with dataframe
    data = pd.merge(suspension, data, on='sample_ID', how='outer')

    # -----SOLIDIFICATION TABLES-----#
    # Main solidification table
    solidification = pd.read_sql('SELECT * FROM solidification', conn)
    solidification.columns = ['solidification_ID', 'sample_ID', 'technique', 'direction', 'vel_to_gravity', 'refrigerant',
                              'cooling_rate', 'temp_cold', 'temp_hot', 'temp_constant', 'gravity', 'velocity', 'temp_nuc',
                              'last_updated']
    solidification = solidification.drop('solidification_ID', axis=1)

    # Mold properties table
    props_mold = pd.read_sql('SELECT * FROM props_mold', conn)
    props_mold.columns = ['props_mold_ID', 'name_mold_mat', 'thermal_cond_mold']
    # Solidification mold table
    solidification_mold = pd.read_sql('SELECT * FROM solid_mold', conn)
    solidification_mold.columns = ['mold_ID', 'sample_ID', 'shape_mold', 'dia_mold', 'height_mold', 'length_mold',
                                   'width_mold', 'wall_mold', 'fillheight', 'props_mold_ID', 'last_update']
    # Merge mold properties table with solidification_mold table
    solidification_mold = pd.merge(props_mold, solidification_mold, on='props_mold_ID', how='outer')
    # Merge solidification_mold table with solidification table
    solidification = pd.merge(solidification_mold, solidification, on='sample_ID', how='outer')
    # Drop rows that are not associated with a sample ID
    solidification = solidification[(solidification['sample_ID']) > 0]
    # Merge solidification table with dataframe
    data = pd.merge(solidification, data, on='sample_ID', how='outer')
    # -----SUBLIMATION TABLE-----#
    sublimation = pd.read_sql('SELECT * FROM sublimation', conn)
    sublimation.columns = ['sublimation_ID', 'sample_ID', 'sublimated', 'time_sub', 'last_updated']
    sublimation = sublimation.drop('last_updated', axis=1)
    sublimation = sublimation.drop('sublimation_ID', axis=1)
    # Merge sublimation table with dataframe
    data = pd.merge(sublimation, data, on='sample_ID', how='outer')
    # -----SINTERING TABLES-----#
    # Sinter1 table
    sinter_1 = pd.read_sql('SELECT * FROM sinter1', conn)
    sinter_1.columns = ['sinter_1_ID', 'sample_ID', 'time_sinter_1', 'temp_sinter_1', 'rampC_sinter_1', 'rampH_sinter_1',
                        'last_updated']
    sinter_1 = sinter_1.drop('last_updated', axis=1)
    sinter_1 = sinter_1.drop('sinter_1_ID', axis=1)
    # Sinter2 table
    sinter_2 = pd.read_sql('SELECT * FROM sinter2', conn)
    sinter_2.columns = ['sinter_2_ID', 'sample_ID', 'time_sinter_2', 'temp_sinter_2', 'rampC_sinter_2',
                        'rampH_sinter_2', 'last_updated']
    sinter_2 = sinter_2.drop('last_updated', axis=1)
    sinter_2 = sinter_2.drop('sinter_2_ID', axis=1)
    # Merge sintering tables, then merge sintering table with samples table
    sinter = pd.merge(sinter_1, sinter_2, on='sample_ID', how='outer')
    data = pd.merge(sinter, data, on='sample_ID', how='outer')

    # Shrinkage table
    shrink = pd.read_sql('SELECT * FROM shrinkage', conn)
    shrink.columns = ['shrinkage_ID', 'sample_ID', 'shrink_vol', 'shrink_dia', 'shrink_lin', 'last_updated']
    shrink = shrink.drop('last_updated', axis=1)
    shrink = shrink.drop('shrinkage_ID', axis=1)
    data = pd.merge(shrink, data, on='sample_ID', how='outer')

    # -----MICROSTRUCTURE TABLES-----#
    microstructure = pd.read_sql('SELECT * FROM microstructure', conn)
    microstructure.columns = ['micro_ID', 'sample_ID', 'pore_structure', 'porosity', 'spacing', 'pore', 'wall',
                              'aspectRatio_pore', 'aspectRatio_wall', 'surface_area', 'last_updated']
    microstructure = microstructure.drop('last_updated', axis=1)
    microstructure = microstructure.drop('micro_ID', axis=1)

    # Merge microstructure table with dataframe
    data = pd.merge(microstructure, data, on='sample_ID', how='outer')

    # -----MECHANICAL TABLES-----#
    # Main mechanical table
    mechanical = pd.read_sql('SELECT * FROM mechanical', conn)
    mechanical.columns = ['mech_ID', 'sample_ID', 'shape_mech', 'height_mech', 'dia_mech', 'length_mech', 'width_mech',
                          'ratio_mech', 'volume_mech', 'compressive', 'flexural', 'elastic', 'strain_rate',
                          'crossheadspeed', 'last_updated']
    mechanical = mechanical.drop('last_updated', axis=1)
    mechanical = mechanical.drop('mech_ID', axis=1)

    # Bulk material properties table
    mech_bulk = pd.read_sql('SELECT * FROM props_mech_bulk', conn)
    mech_bulk.columns = ['material_ID', 'name_material', 'density_material_bulk', 'compressive_bulk', 'elastic_bulk',
                         'flexural_bulk', 'last_updated']

    # Merge bulk mechanical and bulk material properties tables with dataframe
    data = pd.merge(mechanical, data, on='sample_ID', how='outer')
    data = pd.merge(data, mech_bulk, on='material_ID', how='outer')

    # Drop rows that are not associated with a sample ID
    data = data[(data['sample_ID']) > 0]
    # Create columns for normalized mechanical values
    data['norm_compressive'] = (data['compressive'] / data['compressive_bulk'])
    data['norm_flexural'] = (data['flexural'] / data['flexural_bulk'])
    data['norm_elastic'] = (data['elastic'] / data['elastic_bulk'])

    # Drop non-usable columns
    # data.drop(['last_updated_x', 'name_author', 'authors', 'author_ID', 'paper_ID', 'sample_ID',
    #          'title', 'journal', 'volume', 'pages', 'year', 'doi', 'affiliation', 'city',
    #          'props_part_ID1', 'props_fluid_ID1', 'props_part_ID1', 'props_disp1_ID', 'material_ID', 'country', 'long',
    #          'lat', 'email', 'last_updated_y', 'last_update', 'mold_ID'], axis=1, inplace=True)
    # Usable cols
    selected_cols = ['pore_structure', 'name_part1', 'name_fluid1', 'sublimated', 'material', 'technique', 'direction',
                     'material_group',
                     'temp_cold', 'cooling_rate',
                     'time_sub',
                     'time_sinter_1', 'temp_sinter_1', 'vf_part_1', 'vf_fluid_1', 'porosity', 'pore']

    selected_cols.remove('sublimated'), selected_cols.remove('technique'), selected_cols.remove('pore'),
    selected_cols.remove('pore_structure'), selected_cols.remove('direction')
    # 'technique', 'sublimated' only have one value, direction' only has 600 rows. Hence, these columns were dropped.
    # data = data[selected_cols]

    # data.drop('temp_cold', axis=1, inplace=True)  #  Highest corr with porosity of 14%, sadly only has 545 rows

    # Select only rows that porosity is not null
    #data = data[data['porosity'].notna()]
    return data


def filter_str_cols_by_rank(df, min_rank=6):
    str_cols = df.select_dtypes(include=[np.object]).columns
    for col in str_cols:
        rank_filter = df[col].value_counts().head(min_rank).axes[0]  # Filter top 5 in column
        filtered_df = df[df[col].isin(rank_filter)]
        df = filtered_df
    return df


def filter_str_cols_by_rows(df, min_rows=500):
    str_cols = df.select_dtypes(include=[np.object]).columns
    for col in str_cols:
        count_filter = df.groupby(col).filter(lambda x: len(x) > min_rows)[col].unique()
        filtered_df = df[df[col].isin(count_filter)]
        df = filtered_df
    return df


def df_str(df):
    str_cols = df.select_dtypes(include=[np.object]).columns
    df_str = df[str_cols]
    return df_str


