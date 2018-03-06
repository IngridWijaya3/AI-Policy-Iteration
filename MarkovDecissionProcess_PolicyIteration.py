import numpy as np


def valueIteration(states,actions,discountedFactor,actionRewards,probabilityToState,epsilon):
    utitlityValues = np.zeros(len(states))
    policyActionAndUtility = {}
    quit=False

    while (quit==False):
        previousvalue = np.copy(utitlityValues)

        for i in range(len(states) ):
            expectedFututreUtility=np.dot(probabilityToState[i][:][:], utitlityValues)
            actionUtilyValues=actionRewards[i] + discountedFactor*(np.dot(probabilityToState[i][:][:], utitlityValues))
            indexmax=np.argmax(actionUtilyValues)
            action=actions[indexmax]
            utitlityValues[i] = np.max(actionUtilyValues)
            policyActionAndUtility[states[i]]=(action,utitlityValues[i])


        if np.max(np.abs(utitlityValues - previousvalue)) <= epsilon:
            quit=True

    return policyActionAndUtility
def OptimalUtiltyAndAction(discountedFactor = 0.9):
  

    actions = ['use', 'replace']
    states = ['new', 'used1', 'used2', 'used3', 'used4', 'used5', 'used6', 'used7', 'used8', 'dead']

    stateActionDictList = []
    actionRewards = np.zeros((len(states), len(actions)))
    stateIteration = range(len(states))

    probabilityToState = np.zeros((len(states), len(actions), len(states)))
    actionRewards[0][0] = 100  # NEW to USE
    probabilityToState[0, 0, 1] = 1

    for i in stateIteration[1:9]:
        actionRewards[i][0] = 100 - (10 * i)  # USED TO USE
        actionRewards[i][1] = -250  # USED TO REPLACE
        probabilityToState[i, 0, i] = 1 - (0.1 * i)  # USED I
        probabilityToState[i, 0, i + 1] = 0.1 * i  # USED i+1
        probabilityToState[i, 1, 0] = 1

    # USED8

    probabilityToState[9, 1, 0] = 1
    actionRewards[9][1] = -250
    policyActionAndUtility = valueIteration(states, actions, discountedFactor, actionRewards, probabilityToState, 0.001)
    #print(policyActionAndUtility)
    for key, value in policyActionAndUtility.items():
        print(key + " : "+ value[0]+" , "+str(value[1]))

def Used1AndUsed2Price(discountedFactor = 0.9):


    states = ['new', 'used1', 'used2', 'used3', 'used4', 'used5', 'used6', 'used7', 'used8', 'dead']
    actions = ['use', 'replace','replace_to_use']
    stateActionDictList = []
    actionRewards = np.zeros((len(states), len(actions)))
    stateIteration = range(len(states))

    probabilityToState = np.zeros((len(states), len(actions), len(states)))
    actionRewards[0][0] = 100  # NEW to USE
    probabilityToState[0, 0, 1] = 1
    possiblevaluesForUsed3To8={}
    for rewardforR2 in range(-240,1,10):

        for i in stateIteration[1:9]:
            if (i>2 and rewardforR2 == -240):
                possiblevaluesForUsed3To8[states[i]] =[]
            actionRewards[i][0] = 100 - (10 * i)  # USED TO USE
            actionRewards[i][1] = -250  # USED TO REPLACE
            probabilityToState[i, 0, i] = 1 - (0.1 * i)  # USED i
            probabilityToState[i, 0, i + 1] = 0.1 * i  # USED i+1
            if(i>2):
                actionRewards[i][2] = rewardforR2
                probabilityToState[i, 2, 1 ] = 0.5  # USED 1
                probabilityToState[i, 2, 2] = 0.5  # USED 2


            probabilityToState[i, 1, 0] = 1

        probabilityToState[9, 1, 0] = 1
        probabilityToState[9, 2, 1] = 0.5  # USED 1
        probabilityToState[9, 2, 2] = 0.5  # USED 2
        actionRewards[9][1] = -250
        actionRewards[9][2] = rewardforR2


        policyActionAndUtility = valueIteration(states, actions, discountedFactor, actionRewards, probabilityToState, 0.001)
        for usedstateindex in range(3,9,1):
            usedState=states[usedstateindex]
            usedAction=policyActionAndUtility[usedState][0]
            usedUtilityValue=policyActionAndUtility[usedState][1]
            if(usedAction==actions[2]):
                print("Reward for replace to used 1 or used 2 "+ str(rewardforR2))
                for key, value in policyActionAndUtility.items():
                    print(key + " : " + value[0] + " , " + str(value[1]))
                return rewardforR2

def compareBetaAndOptimalPolicy():
    for dicountedFactor in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        print("Discounted Factor "+str(dicountedFactor))
        Question2AAndB(dicountedFactor)
        Question2C(dicountedFactor)
if __name__ == '__main__':
    print(
        "For each of the 10 states, what is the optimal utility (long term expected discounted value) available in that state (i.e., U∗(state))?")
    print(
        "What is the optimal policy that gives you this optimal utility - i.e., in each state, what is the best action to take in that state?")

    OptimalUtiltyAndAction()
    print(
        "New machines are expensive. Suppose you wanted to open a store selling used machines for less - suppose each used machine you sold had an equal chance of being in Used1 and Used2. If you gave these machines away for free, no one would buy New machines and everyone would come to you instead. If you sold them at 250, no one would buy them. What is highest price you could sell your used machines for, while ensuring that (rational) users would buy them?")

    Used1AndUsed2Price()
    print(
        "For different values of β (such that 0 < β < 1), the utility or value of being in certain states will change. However, the optimal policy may not. Compare the optimal policy for β = 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, etc. Is there a policy that is optimal for all sufficiently large β? What do you make of it?")

    compareBetaAndOptimalPolicy()



