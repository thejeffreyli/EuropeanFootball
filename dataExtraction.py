import sqlite3
import pandas as pd
import numpy as np


def createLabel(match):
      
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']
     
    label = pd.DataFrame()
    label.loc[0,'match_api_id'] = match['match_api_id'] 

    # identify match with 2 labels  
    if home_goals > away_goals:
        label.loc[0,'label'] = "WIN"
    else:
        label.loc[0,'label'] = "NOWIN"

    # return label     
    return label.loc[0]



def createFeatLabel(matches, fifa, horizontal = True):

    print("getting match labels") 

    # Merges features and labels into one frame
    features = pd.merge(matches.apply(createLabel, axis = 1), fifa, on = 'match_api_id', how = 'left')
    
    features.to_csv("labels.csv", index=False)     
    
    # return obtained features
    return features



def getRating(match, playerStats):   

    matchID =  match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    
    ratingData = pd.DataFrame()
    names = []
    
    #Loop through all players
    for player in players:   
            
        stats = playerStats[playerStats.player_api_id == match[player]]
        # get latest stats
        currentStats = stats[stats.date < date].sort_values(by = 'date', ascending = False)
        currentStats = currentStats[:1] 
        
        if np.isnan(match[player]) == False: # obtain new rating
            currentStats.reset_index(inplace = True, drop = True)
            overall_rating = pd.Series(currentStats.loc[0, "overall_rating"])
            
        else: # data cleaning
            overall_rating = pd.Series(0)
            
        
        # make the column names
        names.append("{}_overall_rating".format(player))
        
        # combine the rating stats
        ratingData = pd.concat([ratingData, overall_rating], axis = 1)
    
    # make the rating data set
    ratingData.columns = names        
    ratingData['match_api_id'] = matchID
    ratingData.reset_index(inplace = True, drop = True)    

    return ratingData.iloc[0]   


def getGameData(matches, playerStats):
      
    print("getting match data")         
    
    return matches.apply(lambda x :getRating(x, playerStats), axis = 1)



# Connecting to database
path = "C:/Users/hp/Desktop/MLFall20/"  # Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

# Fetching required data tables

# player data
player_data = pd.read_sql("SELECT * FROM Player;", conn)
cols = ['player_api_id', 'player_name']
player_data = player_data.loc[:, cols]   
player_data.to_csv("player_data.csv", index=False) 

# player statistics: feature variables 
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
player_stats_data.to_csv("player_stats_data.csv", index=False) 

# team data
team_data = pd.read_sql("SELECT * FROM Team;", conn)
cols = ['team_api_id', 'team_short_name']
# Range is only for English Premier League (EPL) Teams
team_data = team_data.loc[range(25,59), cols] 
team_data.to_csv("team_data.csv", index=False) 

# player statistics: feature variables 
Team_Attributes = pd.read_sql("SELECT * FROM Team_Attributes;", conn)
Team_Attributes.to_csv("Team_Attributes.csv", index=False) 

# match data
match_data = pd.read_sql("SELECT * FROM Match;", conn)
match_data = match_data.loc[match_data['league_id'] == 1729] # EPL

cols = ['date', 'match_api_id', 'home_team_api_id', 'away_team_api_id',
        'home_team_goal','away_team_goal','home_player_1','home_player_2','home_player_3',
        'home_player_4','home_player_5','home_player_6','home_player_7',
        'home_player_8','home_player_9','home_player_10','home_player_11', 
        'away_player_1','away_player_2','away_player_3','away_player_4', 
        'away_player_5','away_player_6','away_player_7','away_player_8',
        'away_player_9','away_player_10','away_player_11']
match_data = match_data.loc[:, cols]
match_data.to_csv("match_data.csv", index=False) 



" PLAYER STATISTICS "

playerRating = getGameData(match_data, player_stats_data)
featLabels = createFeatLabel(match_data, playerRating)


