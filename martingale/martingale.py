""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Pirashan Ravikumaran		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: pravikumaran3		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 903948218		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
  		  	   		 	   		  		  		    	 		 		   		 		  
def author():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return "pravikumaran3"  # replaced username

def study_group():
    """
    :return: study group members
    :rtype: str
    """
    return "pravikumaran3"
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def gtid():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return 903948218  # replaced GT ID username
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	   		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    result = False  		  	   		 	   		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	   		  		  		    	 		 		   		 		  
        result = True  		  	   		 	   		  		  		    	 		 		   		 		  
    return result

def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    win_prob = 0.4737  # set appropriately to the probability of a win
    # Determine probability for roulette? -> 10/9:1 odds of losing
    np.random.seed(gtid())  # setting the seed for reproducibility based on gtid
    np.set_printoptions(edgeitems=5)

    exp2_bankroll = 256
    exp1_winning = 0
    exp2_winning = 0
    goal_winning = 80
    bet_amount = 1
    episode_spins = 1000
    exp1_episodes = 10
    exp2_episodes = 1000

    # Initializing Numpy arrays here
    exp1_fig1_winnings = np.zeros((exp1_episodes, episode_spins))
    exp1_fig2_winnings = np.zeros((exp2_episodes, episode_spins))
    exp1_fig3_winnings = np.zeros((exp2_episodes, episode_spins))
    exp2_fig4_winnings = np.zeros((exp2_episodes, episode_spins))
    exp2_fig5_winnings = np.zeros((exp2_episodes, episode_spins))

    exp1_success_count = 0

    # Experiment 1 Figure 1
    for episode in range(10):
        exp1_winning = 0
        bet_amount = 1

        # Spin 0 for each episode should have a winnings value of 0
        exp1_fig1_winnings[episode, 0] = exp1_winning

        for spin in range(1, episode_spins + 1):
            if exp1_winning >= goal_winning:
                exp1_fig1_winnings[episode, spin:] = exp1_winning
                exp1_success_count += 1
                break
            if get_spin_result(win_prob):
                exp1_winning += bet_amount
                bet_amount = 1
            else:
                exp1_winning -= bet_amount
                bet_amount *= 2
            exp1_fig1_winnings[episode, spin] = exp1_winning
        # print(exp1_fig1_winnings)
    # print(exp1_success_count)
    #Plotting Results in Experiment 1 Fig 1
    for episode in range(exp1_episodes):
        plt.plot(range(episode_spins), exp1_fig1_winnings[episode], label=f'Episode {episode + 1}')

    # Creating Fig1 Plot
    plt.xlim(0, 300)  # Set x-axis limit
    plt.ylim(-256, 100)  # Set y-axis limit
    plt.xlabel('Number of Spins')
    plt.ylabel('Total Winnings')
    plt.title('Winnings Across 10 Episodes')
    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig('images/exp1_fig1.png')

    # Experiment 1 Figure 2
    exp1_success_count = 0
    for episode in range(1000):
        exp1_fig2_winning = 0
        bet_amount = 1

        # Spin 0 for each episode should have a winnings value of 0
        exp1_fig2_winnings[episode, 0] = exp1_fig2_winning

        for spin in range(1, episode_spins + 1):
            if exp1_fig2_winning >= goal_winning:
                exp1_fig2_winnings[episode, spin:] = exp1_fig2_winning
                exp1_success_count += 1
                break
            if get_spin_result(win_prob):
                exp1_fig2_winning += bet_amount
                bet_amount = 1
            else:
                exp1_fig2_winning -= bet_amount
                bet_amount *= 2
            exp1_fig2_winnings[episode, spin] = exp1_fig2_winning
        # print(exp1_fig2_winnings)
    # print(exp1_success_count)
    mean_winnings = np.mean(exp1_fig2_winnings, axis=0)
    std_dev_winnings = np.std(exp1_fig2_winnings, axis=0)

    # Plotting Results in Experiment 1 Fig 2
    plt.clf() #Clear plot results from before
    plt.plot(range(episode_spins), mean_winnings, label='Mean Winnings')
    plt.plot(range(episode_spins), mean_winnings + std_dev_winnings, color='green', label='Mean + Std Dev')
    plt.plot(range(episode_spins), mean_winnings - std_dev_winnings, color='red', label='Mean - Std Dev')

    # Creating Fig2 Plot
    plt.xlim(0, 300)  # Set x-axis limit
    plt.ylim(-256, 100)  # Set y-axis limit
    plt.xlabel('Number of Spins')
    plt.ylabel('Total Winnings')
    plt.title('Mean Winnings of Spin Rounds Across 1000 Episodes')
    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig('images/exp1_fig2.png')

    #Experiment 1 Figure 3
    for episode in range(1000):
        exp1_fig3_winning = 0
        bet_amount = 1

        # Spin 0 for each episode should have a winnings value of 0
        exp1_fig3_winnings[episode, 0] = exp1_fig3_winning

        for spin in range(1, episode_spins + 1):
            if exp1_fig3_winning >= goal_winning:
                exp1_fig3_winnings[episode, spin:] = exp1_fig3_winning
                break
            if get_spin_result(win_prob):
                exp1_fig3_winning += bet_amount
                bet_amount = 1
            else:
                exp1_fig3_winning -= bet_amount
                bet_amount *= 2
            exp1_fig3_winnings[episode, spin] = exp1_fig3_winning
        # print(exp1_fig3_winnings)
    median_winnings = np.median(exp1_fig3_winnings, axis=0)

    # Plotting Results in Experiment 1 Fig 2
    plt.clf() #Clear plot results from before
    plt.plot(range(episode_spins), median_winnings, label='Median Winnings')
    plt.plot(range(episode_spins), median_winnings + std_dev_winnings, color='green', label='Median + Std Dev')
    plt.plot(range(episode_spins), median_winnings - std_dev_winnings, color='red', label='Median - Std Dev')

    # Creating Fig2 Plot
    plt.xlim(0, 300)  # Set x-axis limit
    plt.ylim(-256, 100)  # Set y-axis limit
    plt.xlabel('Number of Spins')
    plt.ylabel('Total Winnings')
    plt.title('Median Winnings of Spin Rounds Across 1000 Episodes')
    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig('images/exp1_fig3.png')

    #Experiment 2 Figure 4
    exp2_success_count = 0
    for episode in range(1000):
        exp2_fig4_winning = 0
        bet_amount = 1

        # Spin 0 for each episode should have a winnings value of 0
        exp2_fig4_winnings[episode, 0] = exp2_fig4_winning

        for spin in range(1, episode_spins + 1):
            if exp2_fig4_winning >= goal_winning:
                exp2_fig4_winnings[episode, spin:] = exp2_fig4_winning
                exp2_success_count += 1
                break
            if exp2_fig4_winning <= -exp2_bankroll:
                exp2_fig4_winnings[episode, spin:] = exp2_fig4_winning
                break
            if bet_amount > exp2_bankroll + exp2_fig4_winning:
                bet_amount = exp2_bankroll + exp2_fig4_winning
            if get_spin_result(win_prob):
                exp2_fig4_winning += bet_amount
                bet_amount = 1
            else:
                exp2_fig4_winning -= bet_amount
                bet_amount *= 2
            exp2_fig4_winnings[episode, spin] = exp2_fig4_winning
        # print(exp2_fig4_winnings)
    # print(exp2_success_count)
    mean_winnings = np.mean(exp2_fig4_winnings, axis=0)
    std_dev_winnings = np.std(exp2_fig4_winnings, axis=0)
    # print(mean_winnings[-1])

    # Plotting Results in Experiment 2 Fig 4
    plt.clf()  # Clear plot results from before
    plt.plot(range(episode_spins), mean_winnings, label='Mean Winnings')
    plt.plot(range(episode_spins), mean_winnings + std_dev_winnings, color='green', label='Mean + Std Dev')
    plt.plot(range(episode_spins), mean_winnings - std_dev_winnings, color='red', label='Mean - Std Dev')

    # Creating Fig2 Plot
    plt.xlim(0, 300)  # Set x-axis limit
    plt.ylim(-256, 100)  # Set y-axis limit
    plt.xlabel('Number of Spins')
    plt.ylabel('Total Winnings')
    plt.title('Mean Winnings of Spin Rounds Across 1000 Episodes')
    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig('images/exp2_fig4.png')

    #Experiment 2 Figure 5
    for episode in range(1000):
        exp2_fig5_winning = 0
        bet_amount = 1

        # Spin 0 for each episode should have a winnings value of 0
        exp2_fig5_winnings[episode, 0] = exp2_fig5_winning

        for spin in range(1, episode_spins + 1):
            if exp2_fig5_winning >= goal_winning:
                exp2_fig5_winnings[episode, spin:] = exp2_fig5_winning
                break
            if exp2_fig5_winning <= -exp2_bankroll:
                exp2_fig5_winnings[episode, spin:] = exp2_fig5_winning
                break
            if bet_amount > exp2_bankroll + exp2_fig5_winning:
                bet_amount = exp2_bankroll + exp2_fig5_winning
            if get_spin_result(win_prob):
                exp2_fig5_winning += bet_amount
                bet_amount = 1
            else:
                exp2_fig5_winning -= bet_amount
                bet_amount *= 2
            exp2_fig5_winnings[episode, spin] = exp2_fig5_winning
        # print(exp2_fig5_winnings)
    median_winnings = np.median(exp2_fig5_winnings, axis=0)
    std_dev_winnings = np.std(exp2_fig5_winnings, axis=0)

    # Plotting Results in Experiment 2 Fig 4
    plt.clf()  # Clear plot results from before
    plt.plot(range(episode_spins), median_winnings, label='Median Winnings')
    plt.plot(range(episode_spins), median_winnings + std_dev_winnings, color='green', label='Median + Std Dev')
    plt.plot(range(episode_spins), median_winnings - std_dev_winnings, color='red', label='Median - Std Dev')

    # Creating Fig2 Plot
    plt.xlim(0, 300)  # Set x-axis limit
    plt.ylim(-256, 100)  # Set y-axis limit
    plt.xlabel('Number of Spins')
    plt.ylabel('Total Winnings')
    plt.title('Median Winnings of Spin Rounds Across 1000 Episodes')
    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig('images/exp2_fig5.png')
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
