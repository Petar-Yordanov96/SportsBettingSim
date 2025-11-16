import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)    
    df.dropna(inplace=True)   #drops faulty entries without making a copy

    # creates match result column and checks each entire row whether the home team wins/loses/draws
    df['result'] = df.apply(lambda x: 1 if x.home_goals > x.away_goals 
                                      else (-1 if x.home_goals < x.away_goals else 0), axis=1) 

    # encodes teams numerically
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    team_to_id = {team: i for i, team in enumerate(teams)}
    df['home_id'] = df['home_team'].map(team_to_id)
    df['away_id'] = df['away_team'].map(team_to_id)             #maps the names (strings) of the teams to id's (integers) 

    return df, team_to_id
