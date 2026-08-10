# **CHAPTER 18 Reinforcement Learning** 

_Reinforcement learning_ (RL) is one of the most exciting fields of machine learning today, and also one of the oldest. It has been around since the 1950s, producing many interesting applications over the years,<sup>1</sup> particularly in games (e.g., _TD-Gammon_ , a Backgammon-playing program) and in machine control, but seldom making the headline news. However, a revolution took place in 2013, when researchers from a British startup called DeepMind demonstrated a system that could learn to play just about any Atari game from scratch,<sup>2</sup> eventually outperforming humans<sup>3</sup> in most of them, using only raw pixels as inputs and without any prior knowledge of the rules of the games.<sup>4</sup> This was the first of a series of amazing feats, culminating with the victory of their system AlphaGo against Lee Sedol, a legendary professional player of the game of Go, in March 2016 and against Ke Jie, the world champion, in May 2017. No program had ever come close to beating a master of this game, let alone the world champion. Today the whole field of RL is boiling with new ideas, with a wide range of applications. 

So how did DeepMind (bought by Google for over $500 million in 2014) achieve all this? With hindsight it seems rather simple: they applied the power of deep learning to the field of reinforcement learning, and it worked beyond their wildest dreams. In this chapter I will first explain what reinforcement learning is and what it’s good at, 

- 1 For more details, be sure to check out Richard Sutton and Andrew Barto’s book on RL, _Reinforcement Learning: An Introduction_ (MIT Press). 

- 2 Volodymyr Mnih et al., “Playing Atari with Deep Reinforcement Learning”, arXiv preprint arXiv:1312.5602 (2013). 

- 3 Volodymyr Mnih et al., “Human-Level Control Through Deep Reinforcement Learning”, _Nature_ 518 (2015): 529–533. 

- 4 Check out the videos of DeepMind’s system learning to play _Space Invaders_ , _Breakout_ , and other video games at _https://homl.info/dqn3_ . 

**683** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

then present two of the most important techniques in deep reinforcement learning: policy gradients and deep Q-networks, including a discussion of Markov decision processes. Let’s get started! 

## **Learning to Optimize Rewards** 

In reinforcement learning, a software _agent_ makes _observations_ and takes _actions_ within an _environment_ , and in return it receives _rewards_ from the environment. Its objective is to learn to act in a way that will maximize its expected rewards over time. If you don’t mind a bit of anthropomorphism, you can think of positive rewards as pleasure, and negative rewards as pain (the term “reward” is a bit misleading in this case). In short, the agent acts in the environment and learns by trial and error to maximize its pleasure and minimize its pain. 

This is quite a broad setting, which can apply to a wide variety of tasks. Here are a few examples (see Figure 18-1): 

- **a.** The agent can be the program controlling a robot. In this case, the environment is the real world, the agent observes the environment through a set of _sensors_ such as cameras and touch sensors, and its actions consist of sending signals to activate motors. It may be programmed to get positive rewards whenever it approaches the target destination, and negative rewards whenever it wastes time or goes in the wrong direction. 

- **b.** The agent can be the program controlling _Ms. Pac-Man_ . In this case, the environ‐ ment is a simulation of the Atari game, the actions are the nine possible joystick positions (upper left, down, center, and so on), the observations are screenshots, and the rewards are just the game points. 

- **c.** Similarly, the agent can be the program playing a board game such as Go. It only gets a reward if it wins. 

- **d.** The agent does not have to control a physically (or virtually) moving thing. For example, it can be a smart thermostat, getting positive rewards whenever it is close to the target temperature and saves energy, and negative rewards when humans need to tweak the temperature, so the agent must learn to anticipate human needs. 

- **e.** The agent can observe stock market prices and decide how much to buy or sell every second. Rewards are obviously the monetary gains and losses. 

Note that there may not be any positive rewards at all; for example, the agent may move around in a maze, getting a negative reward at every time step, so it had better find the exit as quickly as possible! There are many other examples of tasks to which reinforcement learning is well suited, such as self-driving cars, recommender 

##### **684 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 



<!-- Start of picture text -->
Mo d 2<br><!-- End of picture text -->



<!-- Start of picture text -->
Agent Environment<br>ds Actions<br>a See Rewards &<br>a Fe <3 observations ¢ Ss<br><!-- End of picture text -->



<!-- Start of picture text -->
- Policyspace==———- RReSulltingbehaviors<br>a<br>: a b | WW :<br>: @ @ Lt —— :<br><!-- End of picture text -->

training. For example, you might use a library like PyBullet or MuJoCo for 3D physics simulation. 

_OpenAI Gym_<sup>10</sup> is a toolkit that provides a wide variety of simulated environments (Atari games, board games, 2D and 3D physical simulations, and so on), that you can use to train agents, compare them, or develop new RL algorithms. 

OpenAI Gym is preinstalled on Colab, but it’s an older version, so you’ll need to replace it with the latest one. You also need to install a few of its dependencies. If you are coding on your own machine instead of Colab, and you followed the installation instructions at _https://homl.info/install_ , then you can skip this step; otherwise, enter these commands: 

```
# Only run these commands on Colab or Kaggle!
%pip install -q -U gym
%pip install -q -U gym[classic_control,box2d,atari,accept-rom-license]
```

The first `%pip` command upgrades Gym to the latest version. The `-q` option stands for _quiet_ : it makes the output less verbose. The `-U` option stands for _upgrade_ . The second `%pip` command installs the libraries required to run various kinds of environments. This includes classic environments from _control theory_ –the science of controlling dynamical systems–such as balancing a pole on a cart. It also includes environments based on the Box2D library—a 2D physics engine for games. Lastly, it includes envi‐ ronments based on the Arcade Learning Environment (ALE), which is an emulator for Atari 2600 games. Several Atari game ROMs are downloaded automatically, and by running this code you agree with Atari’s ROM licenses. 

With that, you’re ready to use OpenAI Gym. Let’s import it and make an environment: 

```
importgym
```

```
env=gym.make("CartPole-v1", render_mode="rgb_array")
```

Here, we’ve created a CartPole environment. This is a 2D simulation in which a cart can be accelerated left or right in order to balance a pole placed on top of it (see Figure 18-4). This is a classic control task. 



The `gym.envs.registry` dictionary contains the names and speci‐ fications of all the available environments. 

- 10 OpenAI is an artificial intelligence research company, funded in part by Elon Musk. Its stated goal is to promote and develop friendly AIs that will benefit humanity (rather than exterminate it). 

##### **688 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 



<!-- Start of picture text -->
Angle<br>i?<br>| angular velocity<br>Velocity} 9<br>I I<br>Position<br><!-- End of picture text -->

```
>>> action=1# accelerate right
>>> obs, reward, done, truncated, info=env.step(action)
>>> obs
array([ 0.02727336,  0.18847767,  0.03625453, -0.26141977], dtype=float32)
>>> reward
1.0
>>> done
False
>>> truncated
False
>>> info
{}
```

The `step()` method executes the desired action and returns five values: 

#### `obs` 

This is the new observation. The cart is now moving toward the right ( `obs[1] > 0` ). The pole is still tilted toward the right ( `obs[2] > 0` ), but its angular velocity is now negative ( `obs[3] < 0` ), so it will likely be tilted toward the left after the next step. 

#### `reward` 

In this environment, you get a reward of 1.0 at every step, no matter what you do, so the goal is to keep the episode running for as long as possible. 

#### `done` 

This value will be `True` when the episode is over. This will happen when the pole tilts too much, or goes off the screen, or after 200 steps (in this last case, you have won). After that, the environment must be reset before it can be used again. 

#### `truncated` 

This value will be `True` when an episode is interrupted early, for example by an environment wrapper that imposes a maximum number of steps per episode (see Gym’s documentation for more details on environment wrappers). Some RL algorithms treat truncated episodes differently from episodes finished normally (i.e., when `done` is `True` ), but in this chapter we will treat them identically. 

#### `info` 

This environment-specific dictionary may provide extra information, just like the one returned by the `reset()` method. 



Once you have finished using an environment, you should call its `close()` method to free resources. 

##### **690 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

Let’s hardcode a simple policy that accelerates left when the pole is leaning toward the left and accelerates right when the pole is leaning toward the right. We will run this policy to see the average rewards it gets over 500 episodes: 

```
defbasic_policy(obs):
angle=obs[2]
return0ifangle<0else1
totals= []
forepisodeinrange(500):
episode_rewards=0
obs, info=env.reset(seed=episode)
forstepinrange(200):
action=basic_policy(obs)
obs, reward, done, truncated, info=env.step(action)
episode_rewards+=reward
ifdoneortruncated:
break
totals.append(episode_rewards)
```

This code is self-explanatory. Let’s look at the result: 

```
>>> importnumpyasnp
>>> np.mean(totals), np.std(totals), min(totals), max(totals)
(41.698, 8.389445512070509, 24.0, 63.0)
```

Even with 500 tries, this policy never managed to keep the pole upright for more than 63 consecutive steps. Not great. If you look at the simulation in this chapter’s notebook, you will see that the cart oscillates left and right more and more strongly until the pole tilts too much. Let’s see if a neural network can come up with a better policy. 

## **Neural Network Policies** 

Let’s create a neural network policy. This neural network will take an observation as input, and it will output the action to be executed, just like the policy we hardcoded earlier. More precisely, it will estimate a probability for each action, and then we will select an action randomly, according to the estimated probabilities (see Figure 18-5). In the case of the CartPole environment, there are just two possible actions (left or right), so we only need one output neuron. It will output the probability _p_ of action 0 (left), and of course the probability of action 1 (right) will be 1 – _p_ . For example, if it outputs 0.7, then we will pick action 0 with 70% probability, or action 1 with 30% probability. 

**Neural Network Policies | 691** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 



<!-- Start of picture text -->
O Action<br>et e8 Multinomial sampling<br>@&® Probability ofaction 0 (left)<br>SO OD<br>SSSI<br>x %) X3 X4 Observations<br><!-- End of picture text -->

problem is thus as simple as can be; the observations are noise-free, and they contain the environment’s full state. 

Here is the code to build a basic neural network policy using Keras: 

```
importtensorflowastf
```

```
model=tf.keras.Sequential([
tf.keras.layers.Dense(5, activation="relu"),
tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

We use a `Sequential` model to define the policy network. The number of inputs is the size of the observation space—which in the case of CartPole is 4—and we have just five hidden units because it’s a fairly simple task. Finally, we want to output a single probability—the probability of going left—so we have a single output neuron using the sigmoid activation function. If there were more than two possible actions, there would be one output neuron per action, and we would use the softmax activation function instead. 

OK, we now have a neural network policy that will take observations and output action probabilities. But how do we train it? 

## **Evaluating Actions: The Credit Assignment Problem** 

If we knew what the best action was at each step, we could train the neural network as usual, by minimizing the cross entropy between the estimated probability distri‐ bution and the target probability distribution. It would just be regular supervised learning. However, in reinforcement learning the only guidance the agent gets is through rewards, and rewards are typically sparse and delayed. For example, if the agent manages to balance the pole for 100 steps, how can it know which of the 100 actions it took were good, and which of them were bad? All it knows is that the pole fell after the last action, but surely this last action is not entirely responsible. This is called the _credit assignment problem_ : when the agent gets a reward, it is hard for it to know which actions should get credited (or blamed) for it. Think of a dog that gets rewarded hours after it behaved well; will it understand what it is being rewarded for? 

To tackle this problem, a common strategy is to evaluate an action based on the sum of all the rewards that come after it, usually applying a _discount factor_ , _γ_ (gamma), at each step. This sum of discounted rewards is called the action’s _return_ . Consider the example in Figure 18-6. If an agent decides to go right three times in a row and gets +10 reward after the first step, 0 after the second step, and finally –50 after the third step, then assuming we use a discount factor _γ_ = 0.8, the first action will have a return of 10 + _γ_ × 0 + _γ_<sup>2</sup> × (–50) = –22. If the discount factor is close to 0, then future rewards won’t count for much compared to immediate rewards. Conversely, if the discount factor is close to 1, then rewards far into the future will count almost 

**Evaluating Actions: The Credit Assignment Problem | 693** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 



<!-- Start of picture text -->
cp ®& A<br>Actions: Right Right Right<br>Rewards: +10 0 -50<br>Sum of discounted v v v<br>fututure rewardsds: -22 +——30% -40 «——30% -50<br>¥L Pl<br>Discount ratio<br><!-- End of picture text -->

_REINFORCE algorithms_ , was introduced back in 1992<sup>11</sup> by Ronald Williams. Here is one common variant: 

**1.** First, let the neural network policy play the game several times, and at each step, compute the gradients that would make the chosen action even more likely—but don’t apply these gradients yet. 

**2.** Once you have run several episodes, compute each action’s advantage, using the method described in the previous section. 

**3.** If an action’s advantage is positive, it means that the action was probably good, and you want to apply the gradients computed earlier to make the action even more likely to be chosen in the future. However, if the action’s advantage is nega‐ tive, it means the action was probably bad, and you want to apply the opposite gradients to make this action slightly _less_ likely in the future. The solution is to multiply each gradient vector by the corresponding action’s advantage. 

**4.** Finally, compute the mean of all the resulting gradient vectors, and use it to perform a gradient descent step. 

Let’s use Keras to implement this algorithm. We will train the neural network policy we built earlier so that it learns to balance the pole on the cart. First, we need a function that will play one step. We will pretend for now that whatever action it takes is the right one so that we can compute the loss and its gradients. These gradients will just be saved for a while, and we will modify them later depending on how good or bad the action turned out to be: 

```
defplay_one_step(env, obs, model, loss_fn):
withtf.GradientTape() astape:
left_proba=model(obs[np.newaxis])
action= (tf.random.uniform([1, 1]) >left_proba)
y_target=tf.constant([[1.]]) -tf.cast(action, tf.float32)
loss=tf.reduce_mean(loss_fn(y_target, left_proba))
```

```
grads=tape.gradient(loss, model.trainable_variables)
obs, reward, done, truncated, info=env.step(int(action))
returnobs, reward, done, truncated, grads
```

Let’s walk though this function: 

- Within the `GradientTape` block (see Chapter 12), we start by calling the model, giving it a single observation. We reshape the observation so it becomes a batch containing a single instance, as the model expects a batch. This outputs the probability of going left. 

> 11 Ronald J. Williams, “Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Leaning”, _Machine Learning_ 8 (1992) : 229–256. 

**Policy Gradients | 695** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

- Next, we sample a random float between 0 and 1, and we check whether it is greater than `left_proba` . The `action` will be `False` with probability `left_proba` , or `True` with probability `1 – left_proba` . Once we cast this Boolean to an integer, the action will be 0 (left) or 1 (right) with the appropriate probabilities. 

- We now define the target probability of going left: it is 1 minus the action (cast to a float). If the action is 0 (left), then the target probability of going left will be 1. If the action is 1 (right), then the target probability will be 0. 

- Then we compute the loss using the given loss function, and we use the tape to compute the gradient of the loss with regard to the model’s trainable variables. Again, these gradients will be tweaked later, before we apply them, depending on how good or bad the action turned out to be. 

- Finally, we play the selected action, and we return the new observation, the reward, whether the episode is ended or not, whether it is truncated or not, and of course the gradients that we just computed. 

Now let’s create another function that will rely on the `play_one_step()` function to play multiple episodes, returning all the rewards and gradients for each episode and each step: 

```
defplay_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
all_rewards= []
all_grads= []
forepisodeinrange(n_episodes):
current_rewards= []
current_grads= []
obs, info=env.reset()
forstepinrange(n_max_steps):
obs, reward, done, truncated, grads=play_one_step(
env, obs, model, loss_fn)
current_rewards.append(reward)
current_grads.append(grads)
ifdoneortruncated:
break
```

```
all_rewards.append(current_rewards)
all_grads.append(current_grads)
```

```
returnall_rewards, all_grads
```

This code returns a list of reward lists: one reward list per episode, containing one reward per step. It also returns a list of gradient lists: one gradient list per episode, each containing one tuple of gradients per step and each tuple containing one gradi‐ ent tensor per trainable variable. 

The algorithm will use the `play_multiple_episodes()` function to play the game several times (e.g., 10 times), then it will go back and look at all the rewards, discount them, and normalize them. To do that, we need a couple more functions; the first 

##### **696 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

will compute the sum of future discounted rewards at each step, and the second will normalize all these discounted rewards (i.e., the returns) across many episodes by subtracting the mean and dividing by the standard deviation: 

```
defdiscount_rewards(rewards, discount_factor):
discounted=np.array(rewards)
forstepinrange(len(rewards) -2, -1, -1):
discounted[step] +=discounted[step+1] *discount_factor
returndiscounted
defdiscount_and_normalize_rewards(all_rewards, discount_factor):
all_discounted_rewards= [discount_rewards(rewards, discount_factor)
forrewardsinall_rewards]
flat_rewards=np.concatenate(all_discounted_rewards)
reward_mean=flat_rewards.mean()
reward_std=flat_rewards.std()
return [(discounted_rewards-reward_mean) /reward_std
fordiscounted_rewardsinall_discounted_rewards]
```

Let’s check that this works: 

```
>>> discount_rewards([10, 0, -50], discount_factor=0.8)
array([-22, -40, -50])
>>> discount_and_normalize_rewards([[10, 0, -50], [10, 20]],
... discount_factor=0.8)
...
[array([-0.28435071, -0.86597718, -1.18910299]),
 array([1.26665318, 1.0727777 ])]
```

The call to `discount_rewards()` returns exactly what we expect (see Figure 18-6). You can verify that the function `discount_and_normalize_rewards()` does indeed return the normalized action advantages for each action in both episodes. Notice that the first episode was much worse than the second, so its normalized advantages are all negative; all actions from the first episode would be considered bad, and conversely all actions from the second episode would be considered good. 

We are almost ready to run the algorithm! Now let’s define the hyperparameters. We will run 150 training iterations, playing 10 episodes per iteration, and each episode will last at most 200 steps. We will use a discount factor of 0.95: 

```
n_iterations=150
n_episodes_per_update=10
n_max_steps=200
discount_factor=0.95
```

We also need an optimizer and the loss function. A regular Nadam optimizer with learning rate 0.01 will do just fine, and we will use the binary cross-entropy loss function because we are training a binary classifier (there are two possible actions— left or right): 

```
optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01)
loss_fn=tf.keras.losses.binary_crossentropy
```

**Policy Gradients | 697** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

We are now ready to build and run the training loop! 

```
foriterationinrange(n_iterations):
all_rewards, all_grads=play_multiple_episodes(
env, n_episodes_per_update, n_max_steps, model, loss_fn)
all_final_rewards=discount_and_normalize_rewards(all_rewards,
discount_factor)
all_mean_grads= []
forvar_indexinrange(len(model.trainable_variables)):
mean_grads=tf.reduce_mean(
            [final_reward*all_grads[episode_index][step][var_index]
forepisode_index, final_rewardsinenumerate(all_final_rewards)
forstep, final_rewardinenumerate(final_rewards)], axis=0)
all_mean_grads.append(mean_grads)
```

```
optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
```

Let’s walk through this code: 

- At each training iteration, this loop calls the `play_multiple_episodes()` func‐ tion, which plays 10 episodes and returns the rewards and gradients for each step in each episode. 

- Then we call the `discount_and_normalize_rewards()` function to compute each action’s normalized advantage, called the `final_reward` in this code. This pro‐ vides a measure of how good or bad each action actually was, in hindsight. 

- Next, we go through each trainable variable, and for each of them we compute the weighted mean of the gradients for that variable over all episodes and all steps, weighted by the `final_reward` . 

- Finally, we apply these mean gradients using the optimizer: the model’s trainable variables will be tweaked, and hopefully the policy will be a bit better. 

And we’re done! This code will train the neural network policy, and it will success‐ fully learn to balance the pole on the cart. The mean reward per episode will get very close to 200. By default, that’s the maximum for this environment. Success! 

The simple policy gradients algorithm we just trained solved the CartPole task, but it would not scale well to larger and more complex tasks. Indeed, it is highly _sample inefficient_ , meaning it needs to explore the game for a very long time before it can make significant progress. This is due to the fact that it must run multiple episodes to estimate the advantage of each action, as we have seen. However, it is the foundation of more powerful algorithms, such as _actor-critic_ algorithms (which we will discuss briefly at the end of this chapter). 

**698 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 





<!-- Start of picture text -->
‘QD<br><!-- End of picture text -->



<!-- Start of picture text -->
+40<br>roan hte<br>of 03 i<br><!-- End of picture text -->

Bellman found a way to estimate the _optimal state value_ of any state _s_ , noted _V_ *( _s_ ), which is the sum of all discounted future rewards the agent can expect on average after it reaches the state, assuming it acts optimally. He showed that if the agent acts optimally, then the _Bellman optimality equation_ applies (see Equation 18-1). This recursive equation says that if the agent acts optimally, then the optimal value of the current state is equal to the reward it will get on average after taking one optimal action, plus the expected optimal value of all possible next states that this action can lead to. 

_Equation 18-1. Bellman optimality equation_ 

V * s = max T s , a , s′ R s , a , s′ + γ · V * s′ for all s a<sup>∑</sup> s′ 

In this equation: 

- _T_ ( _s_ , _a_ , _s_ ′ ) is the transition probability from state _s_ to state _s_ ′ , given that the agent chose action _a_ . For example, in Figure 18-8, _T_ ( _s_ 2, _a_ 1, _s_ 0) = 0.8. 

- _R_ ( _s_ , _a_ , _s_ ′ ) is the reward that the agent gets when it goes from state _s_ to state _s_ ′ , given that the agent chose action _a_ . For example, in Figure 18-8, _R_ ( _s_ 2, _a_ 1, _s_ 0) = +40. 

- _γ_ is the discount factor. 

This equation leads directly to an algorithm that can precisely estimate the optimal state value of every possible state: first initialize all the state value estimates to zero, and then iteratively update them using the _value iteration_ algorithm (see Equation 18-2). A remarkable result is that, given enough time, these estimates are guaranteed to converge to the optimal state values, corresponding to the optimal policy. 

_Equation 18-2. Value iteration algorithm_ 

V k + 1 s max T s , a , s′ R s , a , s′ + γ · V k s′ for all s a<sup>∑</sup> s′ 

In this equation, _Vk_ ( _s_ ) is the estimated value of state _s_ at the _k_<sup>th</sup> iteration of the algorithm. 



This algorithm is an example of _dynamic programming_ , which breaks down a complex problem into tractable subproblems that can be tackled iteratively. 

**Markov Decision Processes | 701** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

Knowing the optimal state values can be useful, in particular to evaluate a policy, but it does not give us the optimal policy for the agent. Luckily, Bellman found a very similar algorithm to estimate the optimal _state-action values_ , generally called _Q-values_ (quality values). The optimal Q-value of the state-action pair ( _s_ , _a_ ), noted _Q_ *( _s_ , _a_ ), is the sum of discounted future rewards the agent can expect on average after it reaches the state _s_ and chooses action _a_ , but before it sees the outcome of this action, assuming it acts optimally after that action. 

Let’s look at how it works. Once again, you start by initializing all the Q-value estimates to zero, then you update them using the _Q-value iteration_ algorithm (see Equation 18-3). 

_Equation 18-3. Q-value iteration algorithm_ 



Once you have the optimal Q-values, defining the optimal policy, noted _π_ *( _s_ ), is trivial; when the agent is in state _s_ , it should choose the action with the highest Q-value for that state: π * s = argmax Q * s , a . a 

Let’s apply this algorithm to the MDP represented in Figure 18-8. First, we need to define the MDP: 

```
transition_probabilities= [  # shape=[s, a, s']
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
    [None, [0.8, 0.1, 0.1], None]
]
rewards= [  # shape=[s, a, s']
    [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]
]
possible_actions= [[0, 1, 2], [0, 2], [1]]
```

For example, to know the transition probability of going from _s_ 2 to _s_ 0 after play‐ ing action _a_ 1, we will look up `transition_probabilities[2][1][0]` (which is 0.8). Similarly, to get the corresponding reward, we will look up `rewards[2][1] [0]` (which is +40). And to get the list of possible actions in _s_ 2, we will look up `possible_actions[2]` (in this case, only action _a_ 1 is possible). Next, we must initialize all the Q-values to zero (except for the impossible actions, for which we set the Q-values to –∞): 

```
Q_values=np.full((3, 3), -np.inf)  # -np.inf for impossible actions
forstate, actionsinenumerate(possible_actions):
Q_values[state, actions] =0.0# for all possible actions
```

##### **702 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

Now let’s run the Q-value iteration algorithm. It applies Equation 18-3 repeatedly, to all Q-values, for every state and every possible action: 

```
gamma=0.90# the discount factor
```

```
foriterationinrange(50):
Q_prev=Q_values.copy()
forsinrange(3):
forainpossible_actions[s]:
Q_values[s, a] =np.sum([
transition_probabilities[s][a][sp]
* (rewards[s][a][sp] +gamma*Q_prev[sp].max())
forspinrange(3)])
```

That’s it! The resulting Q-values look like this: 

```
>>> Q_values
array([[18.91891892, 17.02702702, 13.62162162],
       [ 0.        ,        -inf, -4.87971488],
       [       -inf, 50.13365013,        -inf]])
```

For example, when the agent is in state _s_ 0 and it chooses action _a_ 1, the expected sum of discounted future rewards is approximately 17.0. 

For each state, we can find the action that has the highest Q-value: 

```
>>> Q_values.argmax(axis=1)  # optimal action for each state
array([0, 0, 1])
```

This gives us the optimal policy for this MDP when using a discount factor of 0.90: in state _s_ 0 choose action _a_ 0, in state _s_ 1 choose action _a_ 0 (i.e., stay put), and in state _s_ 2 choose action _a_ 1 (the only possible action). Interestingly, if we increase the discount factor to 0.95, the optimal policy changes: in state _s_ 1 the best action becomes _a_ 2 (go through the fire!). This makes sense because the more you value future rewards, the more you are willing to put up with some pain now for the promise of future bliss. 

## **Temporal Difference Learning** 

Reinforcement learning problems with discrete actions can often be modeled as Markov decision processes, but the agent initially has no idea what the transition probabilities are (it does not know _T_ ( _s_ , _a_ , _s_ ′ )), and it does not know what the rewards are going to be either (it does not know _R_ ( _s_ , _a_ , _s_ ′ )). It must experience each state and each transition at least once to know the rewards, and it must experience them multiple times if it is to have a reasonable estimate of the transition probabilities. 

The _temporal difference (TD) learning_ algorithm is very similar to the Q-value itera‐ tion algorithm, but tweaked to take into account the fact that the agent has only partial knowledge of the MDP. In general we assume that the agent initially knows only the possible states and actions, and nothing more. The agent uses an _explora‐ tion policy_ —for example, a purely random policy—to explore the MDP, and as it 

**Temporal Difference Learning | 703** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

progresses, the TD learning algorithm updates the estimates of the state values based on the transitions and rewards that are actually observed (see Equation 18-4). 

_Equation 18-4. TD learning algorithm_ 

V k + 1 s 1 − α V k s + α r + γ · V k s′ or, equivalently: V k + 1 s V k s + α · δk s , r , s′ with δk s , r , s′ = r + γ · V k s′ − V k s 

In this equation: 

- _α_ is the learning rate (e.g., 0.01). 

- _r_ + _γ_ · _Vk_ ( _s_ ′ ) is called the _TD target_ . 

- _δ_ k( _s_ , _r_ , _s_ ′ ) is called the _TD error_ . 

A more concise way of writing the first form of this equation is to use the notation a α<sup>b, which means</sup><sup>_ak_+1 ← (1 –</sup><sup>_α_) ·</sup><sup>_ak_+</sup><sup>_α_·</sup><sup>_bk_. So, the first line of Equation 18-4 can</sup> be rewritten like this: V s α<sup>r+γ·V</sup> s′ . 



TD learning has many similarities with stochastic gradient descent, including the fact that it handles one sample at a time. Moreover, just like SGD, it can only truly converge if you gradually reduce the learning rate; otherwise, it will keep bouncing around the optimum Q-values. 

For each state _s_ , this algorithm keeps track of a running average of the immediate rewards the agent gets upon leaving that state, plus the rewards it expects to get later, assuming it acts optimally. 

## **Q-Learning** 

Similarly, the Q-learning algorithm is an adaptation of the Q-value iteration algo‐ rithm to the situation where the transition probabilities and the rewards are initially unknown (see Equation 18-5). Q-learning works by watching an agent play (e.g., ran‐ domly) and gradually improving its estimates of the Q-values. Once it has accurate Q-value estimates (or close enough), then the optimal policy is just choosing the action that has the highest Q-value (i.e., the greedy policy). 

##### **704 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

_Equation 18-5. Q-learning algorithm_ 



For each state-action pair ( _s_ , _a_ ), this algorithm keeps track of a running average of the rewards _r_ the agent gets upon leaving the state _s_ with action _a_ , plus the sum of discounted future rewards it expects to get. To estimate this sum, we take the ′ maximum of the Q-value estimates for the next state _s_ , since we assume that the target policy will act optimally from then on. 

Let’s implement the Q-learning algorithm. First, we will need to make an agent explore the environment. For this, we need a step function so that the agent can execute one action and get the resulting state and reward: 

```
defstep(state, action):
probas=transition_probabilities[state][action]
next_state=np.random.choice([0, 1, 2], p=probas)
reward=rewards[state][action][next_state]
returnnext_state, reward
```

Now let’s implement the agent’s exploration policy. Since the state space is pretty small, a simple random policy will be sufficient. If we run the algorithm for long enough, the agent will visit every state many times, and it will also try every possible action many times: 

```
defexploration_policy(state):
returnnp.random.choice(possible_actions[state])
```

Next, after we initialize the Q-values just like earlier, we are ready to run the Q- learning algorithm with learning rate decay (using power scheduling, introduced in Chapter 11): 

```
alpha0=0.05# initial learning rate
decay=0.005# learning rate decay
gamma=0.90# discount factor
state=0# initial state
foriterationinrange(10_000):
action=exploration_policy(state)
next_state, reward=step(state, action)
next_value=Q_values[next_state].max()  # greedy policy at the next step
alpha=alpha0/ (1+iteration*decay)
Q_values[state, action] *=1-alpha
Q_values[state, action] +=alpha* (reward+gamma*next_value)
state=next_state
```

**Q-Learning | 705** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 



<!-- Start of picture text -->
Q-Value Iteration Q-Learning<br>20 Po Pt |<br><s<br>aI7  |<br>3 15<br>Ry<br>a}<br>7 10<br>2<br>o<br>0<br>0 10 20 30 40 50 0 2000 4000 6000 8000 10000<br>Iterations Iterations<br><!-- End of picture text -->

quite common to start with a high value for _ε_ (e.g., 1.0) and then gradually reduce it (e.g., down to 0.05). 

Alternatively, rather than relying only on chance for exploration, another approach is to encourage the exploration policy to try actions that it has not tried much before. This can be implemented as a bonus added to the Q-value estimates, as shown in Equation 18-6. 

_Equation 18-6. Q-learning using an exploration function_ 



In this equation: 

- _N_ ( _s_ ′ , _a_ ′ ) counts the number of times the action _a_ ′ was chosen in state _s_ ′ . 

- _f_ ( _Q_ , _N_ ) is an _exploration function_ , such as _f_ ( _Q_ , _N_ ) = _Q_ + _κ_ /(1 + _N_ ), where _κ_ is a curiosity hyperparameter that measures how much the agent is attracted to the unknown. 

### **Approximate Q-Learning and Deep Q-Learning** 

The main problem with Q-learning is that it does not scale well to large (or even medium) MDPs with many states and actions. For example, suppose you wanted to use Q-learning to train an agent to play _Ms. Pac-Man_ (see Figure 18-1). There are about 150 pellets that Ms. Pac-Man can eat, each of which can be present or absent (i.e., already eaten). So, the number of possible states is greater than 2<sup>150</sup> ≈ 10<sup>45</sup> . And if you add all the possible combinations of positions for all the ghosts and Ms. Pac-Man, the number of possible states becomes larger than the number of atoms in our planet, so there’s absolutely no way you can keep track of an estimate for every single Q-value. 

The solution is to find a function _Q_ **θ** ( _s_ , _a_ ) that approximates the Q-value of any stateaction pair ( _s_ , _a_ ) using a manageable number of parameters (given by the parameter vector **θ** ). This is called _approximate Q-learning_ . For years it was recommended to use linear combinations of handcrafted features extracted from the state (e.g., the distances of the closest ghosts, their directions, and so on) to estimate Q-values, but in 2013, DeepMind showed that using deep neural networks can work much better, especially for complex problems, and it does not require any feature engineering. A DNN used to estimate Q-values is called a _deep Q-network_ (DQN), and using a DQN for approximate Q-learning is called _deep Q-learning_ . 

Now, how can we train a DQN? Well, consider the approximate Q-value computed by the DQN for a given state-action pair ( _s_ , _a_ ). Thanks to Bellman, we know we want 

**Q-Learning | 707** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

this approximate Q-value to be as close as possible to the reward _r_ that we actually observe after playing action _a_ in state _s_ , plus the discounted value of playing optimally from then on. To estimate this sum of future discounted rewards, we can just execute ′ ′ the DQN on the next state _s_ , for all possible actions _a_ . We get an approximate future Q-value for each possible action. We then pick the highest (since we assume we will be playing optimally) and discount it, and this gives us an estimate of the sum of future discounted rewards. By summing the reward _r_ and the future discounted value estimate, we get a target Q-value _y_ ( _s_ , _a_ ) for the state-action pair ( _s_ , _a_ ), as shown in Equation 18-7. 

_Equation 18-7. Target Q-value_ 



With this target Q-value, we can run a training step using any gradient descent algorithm. Specifically, we generally try to minimize the squared error between the estimated Q-value _Q_ **θ** ( _s_ , _a_ ) and the target Q-value _y_ ( _s_ , _a_ ), or the Huber loss to reduce the algorithm’s sensitivity to large errors. And that’s the deep Q-learning algorithm! Let’s see how to implement it to solve the CartPole environment. 

## **Implementing Deep Q-Learning** 

The first thing we need is a deep Q-network. In theory, we need a neural net that takes a state-action pair as input, and outputs an approximate Q-value. However, in practice it’s much more efficient to use a neural net that takes only a state as input, and outputs one approximate Q-value for each possible action. To solve the CartPole environment, we do not need a very complicated neural net; a couple of hidden layers will do: 

```
input_shape= [4]  # == env.observation_space.shape
n_outputs=2# == env.action_space.n
```

```
model=tf.keras.Sequential([
tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
tf.keras.layers.Dense(32, activation="elu"),
tf.keras.layers.Dense(n_outputs)
])
```

To select an action using this DQN, we pick the action with the largest predicted Q-value. To ensure that the agent explores the environment, we will use an _ε_ -greedy policy (i.e., we will choose a random action with probability _ε_ ): 

**708 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

```
defepsilon_greedy_policy(state, epsilon=0):
ifnp.random.rand() <epsilon:
returnnp.random.randint(n_outputs)  # random action
else:
Q_values=model.predict(state[np.newaxis], verbose=0)[0]
returnQ_values.argmax()  # optimal action according to the DQN
```

Instead of training the DQN based only on the latest experiences, we will store all experiences in a _replay buffer_ (or _replay memory_ ), and we will sample a random training batch from it at each training iteration. This helps reduce the correlations between the experiences in a training batch, which tremendously helps training. For this, we will just use a double-ended queue ( `deque` ): 

```
fromcollectionsimportdeque
```

```
replay_buffer=deque(maxlen=2000)
```



A _deque_ is a queue elements can be efficiently added to or removed from on both ends. Inserting and deleting items from the ends of the queue is very fast, but random access can be slow when the queue gets long. If you need a very large replay buffer, you should use a circular buffer instead (see the notebook for an implementa‐ tion), or check out DeepMind’s Reverb library. 

Each experience will be composed of six elements: a state _s_ , the action _a_ that the ′ agent took, the resulting reward _r_ , the next state _s_ it reached, a Boolean indicating whether the episode ended at that point ( `done` ), and finally another Boolean indicat‐ ing whether the episode was truncated at that point. We will need a small function to sample a random batch of experiences from the replay buffer. It will return six NumPy arrays corresponding to the six experience elements: 

```
defsample_experiences(batch_size):
indices=np.random.randint(len(replay_buffer), size=batch_size)
batch= [replay_buffer[index] forindexinindices]
return [
np.array([experience[field_index] forexperienceinbatch])
forfield_indexinrange(6)
    ]  # [states, actions, rewards, next_states, dones, truncateds]
```

Let’s also create a function that will play a single step using the _ε_ -greedy policy, then store the resulting experience in the replay buffer: 

```
defplay_one_step(env, state, epsilon):
action=epsilon_greedy_policy(state, epsilon)
next_state, reward, done, truncated, info=env.step(action)
replay_buffer.append((state, action, reward, next_state, done, truncated))
returnnext_state, reward, done, truncated, info
```

**Implementing Deep Q-Learning | 709** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

Finally, let’s create one last function that will sample a batch of experiences from the replay buffer and train the DQN by performing a single gradient descent step on this batch: 

```
batch_size=32
discount_factor=0.95
optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn=tf.keras.losses.mean_squared_error
```

```
deftraining_step(batch_size):
experiences=sample_experiences(batch_size)
states, actions, rewards, next_states, dones, truncateds=experiences
next_Q_values=model.predict(next_states, verbose=0)
max_next_Q_values=next_Q_values.max(axis=1)
runs=1.0- (dones|truncateds)  # episode is not done or truncated
target_Q_values=rewards+runs*discount_factor*max_next_Q_values
target_Q_values=target_Q_values.reshape(-1, 1)
mask=tf.one_hot(actions, n_outputs)
withtf.GradientTape() astape:
all_Q_values=model(states)
Q_values=tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
loss=tf.reduce_mean(loss_fn(target_Q_values, Q_values))
```

```
grads=tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Here’s what’s happening in this code: 

- First we define some hyperparameters, and we create the optimizer and the loss function. 

- Then we create the `training_step()` function. It starts by sampling a batch of experiences, then it uses the DQN to predict the Q-value for each possible action in each experience’s next state. Since we assume that the agent will be playing optimally, we only keep the maximum Q-value for each next state. Next, we use Equation 18-7 to compute the target Q-value for each experience’s state-action pair. 

- We want to use the DQN to compute the Q-value for each experienced stateaction pair, but the DQN will also output the Q-values for the other possible actions, not just for the action that was actually chosen by the agent. So, we need to mask out all the Q-values we do not need. The `tf.one_hot()` function makes it possible to convert an array of action indices into such a mask. For example, if the first three experiences contain actions 1, 1, 0, respectively, then the mask will start with `[[0, 1], [0, 1], [1, 0],...]` . We can then multiply the DQN’s output with this mask, and this will zero out all the Q-values we do not want. We then sum over axis 1 to get rid of all the zeros, keeping only the Q-values of the experienced state-action pairs. This gives us the `Q_values` tensor, containing one predicted Q-value for each experience in the batch. 

##### **710 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 



<!-- Start of picture text -->
sf TIN TTT<br>gio}IT EEE<br>SstIIR<br>8 oofTEE PP<br>ere<br>aot |LMT]|||) | || |<br>os}MMWAE PE EE<br>AO a i<br><!-- End of picture text -->

As you can see, the algorithm took a while to start learning anything, in part because _ε_ was very high at the beginning. Then its progress was erratic: it first reached the max reward around episode 220, but it immediately dropped, then bounced up and down a few times, and soon after it looked like it had finally stabilized near the max reward, at around episode 320, its score again dropped down dramatically. This is called _catastrophic forgetting_ , and it is one of the big problems facing virtually all RL algorithms: as the agent explores the environment, it updates its policy, but what it learns in one part of the environment may break what it learned earlier in other parts of the environment. The experiences are quite correlated, and the learning environment keeps changing—this is not ideal for gradient descent! If you increase the size of the replay buffer, the algorithm will be less subject to this problem. Tuning the learning rate may also help. But the truth is, reinforcement learning is hard: training is often unstable, and you may need to try many hyperparameter values and random seeds before you find a combination that works well. For example, if you try changing the activation function from `"elu"` to `"relu"` , the performance will be much lower. 



Reinforcement learning is notoriously difficult, largely because of the training instabilities and the huge sensitivity to the choice of hyperparameter values and random seeds.<sup>13</sup> As the researcher Andrej Karpathy put it, “[Supervised learning] wants to work. […] RL must be forced to work”. You will need time, patience, persever‐ ance, and perhaps a bit of luck too. This is a major reason RL is not as widely adopted as regular deep learning (e.g., convolutional nets). But there are a few real-world applications, beyond AlphaGo and Atari games: for example, Google uses RL to optimize its datacenter costs, and it is used in some robotics applications, for hyperparameter tuning, and in recommender systems. 

You might wonder why we didn’t plot the loss. It turns out that loss is a poor indicator of the model’s performance. The loss might go down, yet the agent might perform worse (e.g., this can happen when the agent gets stuck in one small region of the environment, and the DQN starts overfitting this region). Conversely, the loss could go up, yet the agent might perform better (e.g., if the DQN was underestimating the Q-values and it starts correctly increasing its predictions, the agent will likely perform better, getting more rewards, but the loss might increase because the DQN also sets the targets, which will be larger too). So, it’s preferable to plot the rewards. 

The basic deep Q-learning algorithm we’ve been using so far would be too unstable to learn to play Atari games. So how did DeepMind do it? Well, they tweaked the algorithm! 

13 A great 2018 post by Alex Irpan nicely lays out RL’s biggest difficulties and limitations. 

##### **712 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

## **Deep Q-Learning Variants** 

Let’s look at a few variants of the deep Q-learning algorithm that can stabilize and speed up training. 

### **Fixed Q-value Targets** 

In the basic deep Q-learning algorithm, the model is used both to make predictions and to set its own targets. This can lead to a situation analogous to a dog chasing its own tail. This feedback loop can make the network unstable: it can diverge, oscillate, freeze, and so on. To solve this problem, in their 2013 paper the DeepMind researchers used two DQNs instead of one: the first is the _online model_ , which learns at each step and is used to move the agent around, and the other is the _target model_ used only to define the targets. The target model is just a clone of the online model: 

```
target=tf.keras.models.clone_model(model)  # clone the model's architecture
target.set_weights(model.get_weights())  # copy the weights
```

Then, in the `training_step()` function, we just need to change one line to use the target model instead of the online model when computing the Q-values of the next states: 

```
next_Q_values=target.predict(next_states, verbose=0)
```

Finally, in the training loop, we must copy the weights of the online model to the target model, at regular intervals (e.g., every 50 episodes): 

```
ifepisode%50==0:
```

```
target.set_weights(model.get_weights())
```

Since the target model is updated much less often than the online model, the Q-value targets are more stable, the feedback loop we discussed earlier is dampened, and its effects are less severe. This approach was one of the DeepMind researchers’ main contributions in their 2013 paper, allowing agents to learn to play Atari games from raw pixels. To stabilize training, they used a tiny learning rate of 0.00025, they updated the target model only every 10,000 steps (instead of 50), and they used a very large replay buffer of 1 million experiences. They decreased `epsilon` very slowly, from 1 to 0.1 in 1 million steps, and they let the algorithm run for 50 million steps. Moreover, their DQN was a deep convolutional net. 

**Deep Q-Learning Variants | 713** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

Now let’s take a look at another DQN variant that managed to beat the state of the art once more. 

### **Double DQN** 

In a 2015 paper,<sup>14</sup> DeepMind researchers tweaked their DQN algorithm, increasing its performance and somewhat stabilizing training. They called this variant _double DQN_ . The update was based on the observation that the target network is prone to overestimating Q-values. Indeed, suppose all actions are equally good: the Q-values estimated by the target model should be identical, but since they are approxima‐ tions, some may be slightly greater than others, by pure chance. The target model will always select the largest Q-value, which will be slightly greater than the mean Q-value, most likely overestimating the true Q-value (a bit like counting the height of the tallest random wave when measuring the depth of a pool). To fix this, the researchers proposed using the online model instead of the target model when select‐ ing the best actions for the next states, and using the target model only to estimate the Q-values for these best actions. Here is the updated `training_step()` function: 

```
deftraining_step(batch_size):
experiences=sample_experiences(batch_size)
states, actions, rewards, next_states, dones, truncateds=experiences
next_Q_values=model.predict(next_states, verbose=0)  # ≠ target.predict()
best_next_actions=next_Q_values.argmax(axis=1)
next_mask=tf.one_hot(best_next_actions, n_outputs).numpy()
max_next_Q_values= (target.predict(next_states, verbose=0) *next_mask
                        ).sum(axis=1)
```

```
    [...]  # the rest is the same as earlier
```

Just a few months later, another improvement to the DQN algorithm was propose; we’ll look at that next. 

### **Prioritized Experience Replay** 

Instead of sampling experiences _uniformly_ from the replay buffer, why not sample important experiences more frequently? This idea is called _importance sampling_ (IS) or _prioritized experience replay_ (PER), and it was introduced in a 2015 paper<sup>15</sup> by DeepMind researchers (once again!). 

> 14 Hado van Hasselt et al., “Deep Reinforcement Learning with Double Q-Learning”, _Proceedings of the 30th AAAI Conference on Artificial Intelligence_ (2015): 2094–2100. 

> 15 Tom Schaul et al., “Prioritized Experience Replay”, arXiv preprint arXiv:1511.05952 (2015). 

##### **714 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

More specifically, experiences are considered “important” if they are likely to lead to fast learning progress. But how can we estimate this? One reasonable approach is to measure the magnitude of the TD error _δ_ = _r_ + _γ_ · _V_ ( _s_ ′ ) – _V_ ( _s_ ). A large TD ′ error indicates that a transition ( _s_ , _a_ , _s_ ) is very surprising, and thus probably worth learning from.<sup>16</sup> When an experience is recorded in the replay buffer, its priority is set to a very large value, to ensure that it gets sampled at least once. However, once it is sampled (and every time it is sampled), the TD error _δ_ is computed, and this experience’s priority is set to _p_ = | _δ_ | (plus a small constant to ensure that every expe‐ rience has a nonzero probability of being sampled). The probability _P_ of sampling an experience with priority _p_ is proportional to _p_<sup>_ζ_</sup> , where _ζ_ is a hyperparameter that controls how greedy we want importance sampling to be: when _ζ_ = 0, we just get uniform sampling, and when _ζ_ = 1, we get full-blown importance sampling. In the paper, the authors used _ζ_ = 0.6, but the optimal value will depend on the task. 

There’s one catch, though: since the samples will be biased toward important expe‐ riences, we must compensate for this bias during training by downweighting the experiences according to their importance, or else the model will just overfit the important experiences. To be clear, we want important experiences to be sampled more often, but this also means we must give them a lower weight during training. To do this, we define each experience’s training weight as _w_ = ( _n P_ )<sup>–</sup><sup>_β_</sup> , where _n_ is the number of experiences in the replay buffer, and _β_ is a hyperparameter that controls how much we want to compensate for the importance sampling bias (0 means not at all, while 1 means entirely). In the paper, the authors used _β_ = 0.4 at the beginning of training and linearly increased it to _β_ = 1 by the end of training. Again, the optimal value will depend on the task, but if you increase one, you will usually want to increase the other as well. 

Now let’s look at one last important variant of the DQN algorithm. 

### **Dueling DQN** 

The _dueling DQN_ algorithm (DDQN, not to be confused with double DQN, although both techniques can easily be combined) was introduced in yet another 2015 paper<sup>17</sup> by DeepMind researchers. To understand how it works, we must first note that the Q-value of a state-action pair ( _s_ , _a_ ) can be expressed as _Q_ ( _s_ , _a_ ) = _V_ ( _s_ ) + _A_ ( _s_ , _a_ ), where _V_ ( _s_ ) is the value of state _s_ and _A_ ( _s_ , _a_ ) is the _advantage_ of taking the action _a_ in state _s_ , compared to all other possible actions in that state. Moreover, the value of a state is equal to the Q-value of the best action _a_<sup>*</sup> for that state (since we assume the optimal 

> 16 It could also just be that the rewards are noisy, in which case there are better methods for estimating an experience’s importance (see the paper for some examples). 

> 17 Ziyu Wang et al., “Dueling Network Architectures for Deep Reinforcement Learning”, arXiv preprint arXiv:1511.06581 (2015). 

**Deep Q-Learning Variants | 715** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

policy will pick the best action), so _V_ ( _s_ ) = _Q_ ( _s_ , _a_<sup>*</sup> ), which implies that _A_ ( _s_ , _a_<sup>*</sup> ) = 0. In a dueling DQN, the model estimates both the value of the state and the advantage of each possible action. Since the best action should have an advantage of 0, the model subtracts the maximum predicted advantage from all predicted advantages. Here is a simple DDQN model, implemented using the functional API: 

```
input_states=tf.keras.layers.Input(shape=[4])
hidden1=tf.keras.layers.Dense(32, activation="elu")(input_states)
hidden2=tf.keras.layers.Dense(32, activation="elu")(hidden1)
state_values=tf.keras.layers.Dense(1)(hidden2)
raw_advantages=tf.keras.layers.Dense(n_outputs)(hidden2)
advantages=raw_advantages-tf.reduce_max(raw_advantages, axis=1,
keepdims=True)
Q_values=state_values+advantages
model=tf.keras.Model(inputs=[input_states], outputs=[Q_values])
```

The rest of the algorithm is just the same as earlier. In fact, you can build a double dueling DQN and combine it with prioritized experience replay! More generally, many RL techniques can be combined, as DeepMind demonstrated in a 2017 paper:<sup>18</sup> the paper’s authors combined six different techniques into an agent called _Rainbow_ , which largely outperformed the state of the art. 

As you can see, deep reinforcement learning is a fast-growing field and there’s much more to discover! 

## **Overview of Some Popular RL Algorithms** 

Before we close this chapter, let’s take a brief look at a few other popular algorithms: 

#### _AlphaGo_<sup>_19_</sup> 

AlphaGo uses a variant of _Monte Carlo tree search_ (MCTS) based on deep neural networks to beat human champions at the game of Go. MCTS was invented in 1949 by Nicholas Metropolis and Stanislaw Ulam. It selects the best move after running many simulations, repeatedly exploring the search tree starting from the current position, and spending more time on the most promising branches. When it reaches a node that it hasn’t visited before, it plays randomly until the game ends, and updates its estimates for each visited node (excluding the random moves), increasing or decreasing each estimate depending on the final outcome. AlphaGo is based on the same principle, but it uses a policy network to select moves, rather than playing randomly. This policy net is trained using policy gradients. The original algorithm involved three more neural networks, 

> 18 Matteo Hessel et al., “Rainbow: Combining Improvements in Deep Reinforcement Learning”, arXiv preprint arXiv:1710.02298 (2017): 3215–3222. 

> 19 David Silver et al., “Mastering the Game of Go with Deep Neural Networks and Tree Search”, _Nature_ 529 (2016): 484–489. 

##### **716 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

and was more complicated, but it was simplified in the AlphaGo Zero paper,<sup>20</sup> which uses a single neural network to both select moves and evaluate game states. The AlphaZero paper<sup>21</sup> generalized this algorithm, making it capable of tackling not only the game of Go, but also chess and shogi (Japanese chess). Lastly, the MuZero paper<sup>22</sup> continued to improve upon this algorithm, outperforming the previous iterations even though the agent starts out without even knowing the rules of the game! 

#### _Actor-critic algorithms_ 

Actor-critics are a family of RL algorithms that combine policy gradients with deep Q-networks. An actor-critic agent contains two neural networks: a policy net and a DQN. The DQN is trained normally, by learning from the agent’s experiences. The policy net learns differently (and much faster) than in regular PG: instead of estimating the value of each action by going through multiple episodes, then summing the future discounted rewards for each action, and finally normalizing them, the agent (actor) relies on the action values estimated by the DQN (critic). It’s a bit like an athlete (the agent) learning with the help of a coach (the DQN). 

#### _Asynchronous advantage actor-critic (A3C)_<sup>_23_</sup> 

This is an important actor-critic variant introduced by DeepMind researchers in 2016 where multiple agents learn in parallel, exploring different copies of the environment. At regular intervals, but asynchronously (hence the name), each agent pushes some weight updates to a master network, then it pulls the latest weights from that network. Each agent thus contributes to improving the master network and benefits from what the other agents have learned. Moreover, instead of estimating the Q-values, the DQN estimates the advantage of each action (hence the second A in the name), which stabilizes training. 

#### _Advantage actor-critic (A2C)_ 

A2C is a variant of the A3C algorithm that removes the asynchronicity. All model updates are synchronous, so gradient updates are performed over larger batches, which allows the model to better utilize the power of the GPU. 

- 20 David Silver et al., “Mastering the Game of Go Without Human Knowledge”, _Nature_ 550 (2017): 354–359. 

- 21 David Silver et al., “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algo‐ rithm”, arXiv preprint arXiv:1712.01815. 

- 22 Julian Schrittwieser et al., “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model”, arXiv preprint arXiv:1911.08265 (2019). 

- 23 Volodymyr Mnih et al., “Asynchronous Methods for Deep Reinforcement Learning”, _Proceedings of the 33rd International Conference on Machine Learning_ (2016): 1928–1937. 

**Overview of Some Popular RL Algorithms | 717** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

#### _Soft actor-critic (SAC)_<sup>_24_</sup> 

SAC is an actor-critic variant proposed in 2018 by Tuomas Haarnoja and other UC Berkeley researchers. It learns not only rewards, but also to maximize the entropy of its actions. In other words, it tries to be as unpredictable as possible while still getting as many rewards as possible. This encourages the agent to explore the environment, which speeds up training, and makes it less likely to repeatedly execute the same action when the DQN produces imperfect estimates. This algorithm has demonstrated an amazing sample efficiency (contrary to all the previous algorithms, which learn very slowly). 

#### _Proximal policy optimization (PPO)_<sup>_25_</sup> 

This algorithm by John Schulman and other OpenAI researchers is based on A2C, but it clips the loss function to avoid excessively large weight updates (which often lead to training instabilities). PPO is a simplification of the previous _trust region policy optimization_<sup>26</sup> (TRPO) algorithm, also by OpenAI. OpenAI made the news in April 2019 with its AI called OpenAI Five, based on the PPO algorithm, which defeated the world champions at the multiplayer game _Dota 2_ . 

#### _Curiosity-based exploration_<sup>_27_</sup> 

A recurring problem in RL is the sparsity of the rewards, which makes learning very slow and inefficient. Deepak Pathak and other UC Berkeley researchers have proposed an exciting way to tackle this issue: why not ignore the rewards, and just make the agent extremely curious to explore the environment? The rewards thus become intrinsic to the agent, rather than coming from the environment. Similarly, stimulating curiosity in a child is more likely to give good results than purely rewarding the child for getting good grades. How does this work? The agent continuously tries to predict the outcome of its actions, and it seeks situations where the outcome does not match its predictions. In other words, it wants to be surprised. If the outcome is predictable (boring), it goes elsewhere. However, if the outcome is unpredictable but the agent notices that it has no control over it, it also gets bored after a while. With only curiosity, the authors succeeded in training an agent at many video games: even though the agent gets no penalty for losing, the game starts over, which is boring so it learns to avoid it. 

> 24 Tuomas Haarnoja et al., “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”, _Proceedings of the 35th International Conference on Machine Learning_ (2018): 1856–1865. 

> 25 John Schulman et al., “Proximal Policy Optimization Algorithms”, arXiv preprint arXiv:1707.06347 (2017). 

> 26 John Schulman et al., “Trust Region Policy Optimization”, _Proceedings of the 32nd International Conference on Machine Learning_ (2015): 1889–1897. 

- 27 Deepak Pathak et al., “Curiosity-Driven Exploration by Self-Supervised Prediction”, _Proceedings of the 34th International Conference on Machine Learning_ (2017): 2778–2787. 

##### **718 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

#### _Open-ended learning (OEL)_ 

The objective of OEL is to train agents capable of endlessly learning new and interesting tasks, typically generated procedurally. We’re not there yet, but there has been some amazing progress over the last few years. For example, a 2019 paper<sup>28</sup> by a team of researchers from Uber AI introduced the _POET algorithm_ , which generates multiple simulated 2D environments with bumps and holes and trains one agent per environment: the agent’s goal is to walk as fast as possible while avoiding the obstacles. The algorithm starts out with simple envi‐ ronments, but they gradually get harder over time: this is called _curriculum learning_ . Moreover, although each agent is only trained within one environment, it must regularly compete against other agents, across all environments. In each environment, the winner is copied over and it replaces the agent that was there before. This way, knowledge is regularly transferred across environments, and the most adaptable agents are selected. In the end, the agents are much better walkers than agents trained on a single task, and they can tackle much harder environments. Of course, this principle can be applied to other environments and tasks as well. If you’re interested in OEL, make sure to check out the Enhanced POET paper,<sup>29</sup> as well as DeepMind’s 2021 paper<sup>30</sup> on this topic. 



If you’d like to learn more about reinforcement learning, check out the book _Reinforcement Learning_ by Phil Winder (O’Reilly). 

We covered many topics in this chapter: policy gradients, Markov chains, Markov decision processes, Q-learning, approximate Q-learning, and deep Q-learning and its main variants (fixed Q-value targets, double DQN, dueling DQN, and prioritized experience replay), and finally we took a quick look at a few other popular algo‐ rithms. Reinforcement learning is a huge and exciting field, with new ideas and algorithms popping out every day, so I hope this chapter sparked your curiosity: there is a whole world to explore! 

- 28 Rui Wang et al., “Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions”, arXiv preprint arXiv:1901.01753 (2019). 

- 29 Rui Wang et al., “Enhanced POET: Open-Ended Reinforcement Learning Through Unbounded Invention of Learning Challenges and Their Solutions”, arXiv preprint arXiv:2003.08536 (2020). 

- 30 Open-Ended Learning Team et al., “Open-Ended Learning Leads to Generally Capable Agents”, arXiv preprint arXiv:2107.12808 (2021). 

**Overview of Some Popular RL Algorithms | 719** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

## **Exercises** 

**1.** How would you define reinforcement learning? How is it different from regular supervised or unsupervised learning? 

**2.** Can you think of three possible applications of RL that were not mentioned in this chapter? For each of them, what is the environment? What is the agent? What are some possible actions? What are the rewards? 

**3.** What is the discount factor? Can the optimal policy change if you modify the discount factor? 

**4.** How do you measure the performance of a reinforcement learning agent? 

**5.** What is the credit assignment problem? When does it occur? How can you alleviate it? 

**6.** What is the point of using a replay buffer? 

**7.** What is an off-policy RL algorithm? 

**8.** Use policy gradients to solve OpenAI Gym’s LunarLander-v2 environment. 

**9.** Use a double dueling DQN to train an agent that can achieve a superhuman level at the famous Atari _Breakout_ game ( `"ALE/Breakout-v5"` ). The observations are images. To simplify the task, you should convert them to grayscale (i.e., average over the channels axis) then crop them and downsample them, so they’re just large enough to play, but not more. An individual image does not tell you which way the ball and the paddles are going, so you should merge two or three consecutive images to form each state. Lastly, the DQN should be composed mostly of convolutional layers. 

**10.** If you have about $100 to spare, you can purchase a Raspberry Pi 3 plus some cheap robotics components, install TensorFlow on the Pi, and go wild! For an example, check out this fun post by Lukas Biewald, or take a look at GoPiGo or BrickPi. Start with simple goals, like making the robot turn around to find the brightest angle (if it has a light sensor) or the closest object (if it has a sonar sensor), and move in that direction. Then you can start using deep learning: for example, if the robot has a camera, you can try to implement an object detection algorithm so it detects people and moves toward them. You can also try to use RL to make the agent learn on its own how to use the motors to achieve that goal. Have fun! 

Solutions to these exercises are available at the end of this chapter’s notebook, at _https://homl.info/colab3_ . 

##### **720 | Chapter 18: Reinforcement Learning** 

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O'Reilly Media, Incorporated, 2022. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/uwa/detail.action?docID=30168989. Created from uwa on 2026-03-06 01:18:37. 

