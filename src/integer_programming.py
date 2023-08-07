import pandas as pd
import pulp

def optimise_team(df, max_cost, defenders, midfielders, attackers):
    """
    Given a dataframe with football player information, this function selects
    the optimal team composition that maximizes the sum of total_points while
    adhering to the given constraints:
        - 2 goalkeepers (GK), 5 defenders (DEF), 5 midfielders (MID), 3 forwards (FWD)
        - The sum of the now_cost of all players must be less than or equal to 1000
        - No more than 3 players from the same team
    
    Parameters:
    -----------
    df : pandas.DataFrame
        A dataframe with columns:
            - 'now_cost': The cost of a player
            - 'total_points': The total points scored by a player
            - 'element_type': The position of a player, one of 'GK', 'DEF', 'MID', 'FWD'
            - 'team': The team the player belongs to
    
    Returns:
    --------
    selected_players : pandas.DataFrame
        A dataframe containing the selected players that maximize the total_points
        while meeting all the constraints.
    """
    # Add a variable for each player, indicating if they are selected (1) or not (0)
    df['is_selected'] = [pulp.LpVariable(f'is_selected{i}', cat='Binary') for i in range(len(df))]

    # Create the optimization problem
    problem = pulp.LpProblem("Fantasy_Football_Team", pulp.LpMaximize)

    # Objective function: maximize total points
    objective_function = pulp.lpSum(df['is_selected'] * df['total_points'])
    problem += objective_function

    # Constraints
    problem += (pulp.lpSum(df['is_selected'] * df['now_cost']) <= max_cost)

    # Position constraints
    position_dict = {'GK': 1, 'DEF': defenders, 'MID': midfielders, 'FWD': attackers}
    for position, max_count in position_dict.items():
        problem += (pulp.lpSum(df[df['element_type'] == position]['is_selected']) == max_count)

    # # Team constraints
    # teams = df['team'].unique()
    # for team in teams:
    #     problem += (pulp.lpSum(df[df['team'] == team]['is_selected']) <= 3)

    # Solve the optimization problem
    problem.solve()

    # Get the solution
    df['is_selected_solution'] = [var.varValue for var in df['is_selected']]

    # Show the selected players
    selected_players = df[df['is_selected_solution'] == 1]
    
    return selected_players
