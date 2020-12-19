import os
import pandas as pd

# os.chdir("C:\\Users\\HaohanShi\\Desktop\\FALL 2020\\CS 334\\proj")
os.chdir("C:/Users/hp/Desktop/MLFall20/draft")

# read team attributes and team data
team_att = pd.read_csv("Team_Attributes.csv")
team_dat = pd.read_csv("team_data.csv")


# only selecting team attributes of teams from the EPL
team_ids = team_dat['team_api_id']
eng_teams = team_att.loc[team_att["team_api_id"].isin(team_ids)]
latest_eng_att = eng_teams[eng_teams.groupby('team_api_id')['date'].transform('max') == eng_teams['date']]
latest_eng_att.to_csv("latest_team_attributs.csv", index = False)

# remove columns with repetitive features
dat = latest_eng_att.drop(['id', 'buildUpPlaySpeedClass', 'buildUpPlayPositioningClass', 
                           'chanceCreationCrossingClass', 'chanceCreationShootingClass', 'chanceCreationPositioningClass',
                           'defencePressureClass', 'defenceAggressionClass', 'defenceAggressionClass',
                           'defenceTeamWidthClass', 'defenceDefenderLineClass'], axis = 1) 

# replaces categorial data with numerical data
dat = dat.replace(["Little","Normal","Short","Mixed","Long","Safe","Normal","Risky"],[1,2,1,2,3,1,2,3])
dat.to_csv("latest_team_attributes_numeric.csv", index = False)


add_dat = dat.drop(['team_fifa_api_id', 'date'], axis = 1) 

original_dat = pd.read_csv("labels.csv")
match = pd.read_csv("match_data.csv")


original_ids = original_dat["match_api_id"]
match = match.loc[match["match_api_id"].isin(original_ids)]

home = pd.DataFrame()
away = pd.DataFrame()

for i in range(len(match)):
    away_id = match.iloc[i]["away_team_api_id"]
    home_id = match.iloc[i]["home_team_api_id"]
    away_add = add_dat.loc[add_dat["team_api_id"] == away_id]
    home_add = add_dat.loc[add_dat["team_api_id"] == home_id]
    home = home.append(home_add)
    away = away.append(away_add)

home.columns = ["home_team_api_id", "home_buildUpPlaySpeed",
                "home_buildUpPlayDribbling","home_buildUpPlayDribblingClass",
                "home_buildUpPlayPassing","home_buildUpPlayPassingClass",
                "home_chanceCreationPassing","home_chanceCreationPassingClass",
                "home_chanceCreationCrossing", "home_chanceCreationShooting",
                "home_defencePressure", "home_defenceAggression",
                "home_defenceTeamWidth"]

# home and away attributes
away.to_csv("away.csv", index = False)
home.to_csv("home.csv", index = False)
