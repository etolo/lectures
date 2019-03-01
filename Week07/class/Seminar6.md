# Seminar 6: Getting started with OpenAI Gym

## Introduction to OpenAI Gym
The gym library developed by OpenAI is a collection of test problems — **environments** — that you can use to test your reinforcement learning algorithms.

## Available environments 
Environments available in the gym include classic control, algorithmic, Atari and 2D and 3D robots tasks, from easy to difficult level.  

### Classic control 

Classic control problems from RL literature.  
![Cartpole](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/Cart-Pole.mp4)
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The actions one could take are pushing the cart to the left or right. The goal is to prevent the pendulum from falling over.  A reward of +1 is given for every step that the pendulum remains upright. 

![Mountain Car](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/mountain_car.mp4)
A car is stuck in a valley. The goal is to drive up the mountain and reach where the flat is.  However, the car's engine is not strong enough climb to the top in a single run. The way to succeed is to drive back and forth to build up momentum.  The actions in the problem is also driving the car to the left or right.  A reward of -1 is given for every step until reaching the goal.  

![Acrobot](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/acrobot.mp4)
The acrobot system includes two joints and two links, where the joint between the two links is actuated. Initially, the links are hanging downwards, and the goal is to swing the end of the lower link up to a given height.

### Atari
Teach the agent to learn to play Atari games
![Atari game](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/atari1.mp4)
![Boxing](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/boxing.mp4)

### MoJoCo
Continuous control tasks, running in a fast physics simulator. 
![Ant walking](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/ant_v2.mp4)
Make a four-legged creature walk as fast as possible. 

![Learning to walk](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/Learning_to_walk.mp4)
Make a three-dimensional bipedal robot walk forward as fast as possible. [This link](https://youtu.be/07eIebEk1MM) gives you an idea about the goal we try to achieve.   

### Robotics
Simulated goal-based tasks for the Fetch and ShadowHand robots.

![Fetch, pick and place](hhttps://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/fetch_pick_place.mp4)
A goal is randomly chosen in 3D space. Control Fetch's end effector to grasp and lift the block up to reach that goal.

![robotics](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/robotics.mp4)
A goal orientation is randomly chosen for a block which is placed in a robot hand. The end goal is to control the robot hand to reach the given target orientation for the block.

### Gym Retro
Newly released platform for RL research on games. 
![Racing](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/ferrari.mp4)
![Gradius](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/gradius3.mp4)

![Various games available](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/multi_games.mp4)

To check the list of environments available in your installation, 
```
from gym import envs 
print(envs.registry.all())
```
which returns
```
dict_values([EnvSpec(Copy-v0), EnvSpec(RepeatCopy-v0), EnvSpec(ReversedAddition-v0), EnvSpec(ReversedAddition3-v0), EnvSpec(DuplicatedInput-v0), EnvSpec(Reverse-v0), EnvSpec(CartPole-v0), EnvSpec(CartPole-v1), EnvSpec(MountainCar-v0), EnvSpec(MountainCarContinuous-v0), EnvSpec(Pendulum-v0), EnvSpec(Acrobot-v1), EnvSpec(LunarLander-v2), EnvSpec(LunarLanderContinuous-v2), EnvSpec(BipedalWalker-v2), EnvSpec(BipedalWalkerHardcore-v2), EnvSpec(CarRacing-v0), EnvSpec(Blackjack-v0), EnvSpec(KellyCoinflip-v0), EnvSpec(KellyCoinflipGeneralized-v0), EnvSpec(FrozenLake-v0), EnvSpec(FrozenLake8x8-v0), EnvSpec(CliffWalking-v0), EnvSpec(NChain-v0), EnvSpec(Roulette-v0), EnvSpec(Taxi-v2), EnvSpec(GuessingGame-v0), EnvSpec(HotterColder-v0), EnvSpec(Reacher-v2), EnvSpec(Pusher-v2), EnvSpec(Thrower-v2), EnvSpec(Striker-v2), EnvSpec(InvertedPendulum-v2), EnvSpec(InvertedDoublePendulum-v2), EnvSpec(HalfCheetah-v2), EnvSpec(Hopper-v2), EnvSpec(Swimmer-v2), EnvSpec(Walker2d-v2), EnvSpec(Ant-v2), EnvSpec(Humanoid-v2), EnvSpec(HumanoidStandup-v2), EnvSpec(FetchSlide-v1), EnvSpec(FetchPickAndPlace-v1), EnvSpec(FetchReach-v1), EnvSpec(FetchPush-v1), EnvSpec(HandReach-v0), EnvSpec(HandManipulateBlockRotateZ-v0), EnvSpec(HandManipulateBlockRotateParallel-v0), EnvSpec(HandManipulateBlockRotateXYZ-v0), EnvSpec(HandManipulateBlockFull-v0), EnvSpec(HandManipulateBlock-v0), EnvSpec(HandManipulateEggRotate-v0), EnvSpec(HandManipulateEggFull-v0), EnvSpec(HandManipulateEgg-v0), EnvSpec(HandManipulatePenRotate-v0), EnvSpec(HandManipulatePenFull-v0), EnvSpec(HandManipulatePen-v0), EnvSpec(FetchSlideDense-v1), EnvSpec(FetchPickAndPlaceDense-v1), EnvSpec(FetchReachDense-v1), EnvSpec(FetchPushDense-v1), EnvSpec(HandReachDense-v0), EnvSpec(HandManipulateBlockRotateZDense-v0), EnvSpec(HandManipulateBlockRotateParallelDense-v0), EnvSpec(HandManipulateBlockRotateXYZDense-v0), EnvSpec(HandManipulateBlockFullDense-v0), EnvSpec(HandManipulateBlockDense-v0), EnvSpec(HandManipulateEggRotateDense-v0), EnvSpec(HandManipulateEggFullDense-v0), EnvSpec(HandManipulateEggDense-v0), EnvSpec(HandManipulatePenRotateDense-v0), EnvSpec(HandManipulatePenFullDense-v0), EnvSpec(HandManipulatePenDense-v0), EnvSpec(AirRaid-v0), EnvSpec(AirRaid-v4), EnvSpec(AirRaidDeterministic-v0), EnvSpec(AirRaidDeterministic-v4), EnvSpec(AirRaidNoFrameskip-v0), EnvSpec(AirRaidNoFrameskip-v4), EnvSpec(AirRaid-ram-v0), EnvSpec(AirRaid-ram-v4), EnvSpec(AirRaid-ramDeterministic-v0), EnvSpec(AirRaid-ramDeterministic-v4), EnvSpec(AirRaid-ramNoFrameskip-v0), EnvSpec(AirRaid-ramNoFrameskip-v4), EnvSpec(Alien-v0), EnvSpec(Alien-v4), EnvSpec(AlienDeterministic-v0), EnvSpec(AlienDeterministic-v4), EnvSpec(AlienNoFrameskip-v0), EnvSpec(AlienNoFrameskip-v4), EnvSpec(Alien-ram-v0), EnvSpec(Alien-ram-v4), EnvSpec(Alien-ramDeterministic-v0), EnvSpec(Alien-ramDeterministic-v4), EnvSpec(Alien-ramNoFrameskip-v0), EnvSpec(Alien-ramNoFrameskip-v4), EnvSpec(Amidar-v0), EnvSpec(Amidar-v4), EnvSpec(AmidarDeterministic-v0), EnvSpec(AmidarDeterministic-v4), EnvSpec(AmidarNoFrameskip-v0), EnvSpec(AmidarNoFrameskip-v4), EnvSpec(Amidar-ram-v0), EnvSpec(Amidar-ram-v4), EnvSpec(Amidar-ramDeterministic-v0), EnvSpec(Amidar-ramDeterministic-v4), EnvSpec(Amidar-ramNoFrameskip-v0), EnvSpec(Amidar-ramNoFrameskip-v4), EnvSpec(Assault-v0), EnvSpec(Assault-v4), EnvSpec(AssaultDeterministic-v0), EnvSpec(AssaultDeterministic-v4), EnvSpec(AssaultNoFrameskip-v0), EnvSpec(AssaultNoFrameskip-v4), EnvSpec(Assault-ram-v0), EnvSpec(Assault-ram-v4), EnvSpec(Assault-ramDeterministic-v0), EnvSpec(Assault-ramDeterministic-v4), EnvSpec(Assault-ramNoFrameskip-v0), EnvSpec(Assault-ramNoFrameskip-v4), EnvSpec(Asterix-v0), EnvSpec(Asterix-v4), EnvSpec(AsterixDeterministic-v0), EnvSpec(AsterixDeterministic-v4), EnvSpec(AsterixNoFrameskip-v0), EnvSpec(AsterixNoFrameskip-v4), EnvSpec(Asterix-ram-v0), EnvSpec(Asterix-ram-v4), EnvSpec(Asterix-ramDeterministic-v0), EnvSpec(Asterix-ramDeterministic-v4), EnvSpec(Asterix-ramNoFrameskip-v0), EnvSpec(Asterix-ramNoFrameskip-v4), EnvSpec(Asteroids-v0), EnvSpec(Asteroids-v4), EnvSpec(AsteroidsDeterministic-v0), EnvSpec(AsteroidsDeterministic-v4), EnvSpec(AsteroidsNoFrameskip-v0), EnvSpec(AsteroidsNoFrameskip-v4), EnvSpec(Asteroids-ram-v0), EnvSpec(Asteroids-ram-v4), EnvSpec(Asteroids-ramDeterministic-v0), EnvSpec(Asteroids-ramDeterministic-v4), EnvSpec(Asteroids-ramNoFrameskip-v0), EnvSpec(Asteroids-ramNoFrameskip-v4), EnvSpec(Atlantis-v0), EnvSpec(Atlantis-v4), EnvSpec(AtlantisDeterministic-v0), EnvSpec(AtlantisDeterministic-v4), EnvSpec(AtlantisNoFrameskip-v0), EnvSpec(AtlantisNoFrameskip-v4), EnvSpec(Atlantis-ram-v0), EnvSpec(Atlantis-ram-v4), EnvSpec(Atlantis-ramDeterministic-v0), EnvSpec(Atlantis-ramDeterministic-v4), EnvSpec(Atlantis-ramNoFrameskip-v0), EnvSpec(Atlantis-ramNoFrameskip-v4), EnvSpec(BankHeist-v0), EnvSpec(BankHeist-v4), EnvSpec(BankHeistDeterministic-v0), EnvSpec(BankHeistDeterministic-v4), EnvSpec(BankHeistNoFrameskip-v0), EnvSpec(BankHeistNoFrameskip-v4), EnvSpec(BankHeist-ram-v0), EnvSpec(BankHeist-ram-v4), EnvSpec(BankHeist-ramDeterministic-v0), EnvSpec(BankHeist-ramDeterministic-v4), EnvSpec(BankHeist-ramNoFrameskip-v0), EnvSpec(BankHeist-ramNoFrameskip-v4), EnvSpec(BattleZone-v0), EnvSpec(BattleZone-v4), EnvSpec(BattleZoneDeterministic-v0), EnvSpec(BattleZoneDeterministic-v4), EnvSpec(BattleZoneNoFrameskip-v0), EnvSpec(BattleZoneNoFrameskip-v4), EnvSpec(BattleZone-ram-v0), EnvSpec(BattleZone-ram-v4), EnvSpec(BattleZone-ramDeterministic-v0), EnvSpec(BattleZone-ramDeterministic-v4), EnvSpec(BattleZone-ramNoFrameskip-v0), EnvSpec(BattleZone-ramNoFrameskip-v4), EnvSpec(BeamRider-v0), EnvSpec(BeamRider-v4), EnvSpec(BeamRiderDeterministic-v0), EnvSpec(BeamRiderDeterministic-v4), EnvSpec(BeamRiderNoFrameskip-v0), EnvSpec(BeamRiderNoFrameskip-v4), EnvSpec(BeamRider-ram-v0), EnvSpec(BeamRider-ram-v4), EnvSpec(BeamRider-ramDeterministic-v0), EnvSpec(BeamRider-ramDeterministic-v4), EnvSpec(BeamRider-ramNoFrameskip-v0), EnvSpec(BeamRider-ramNoFrameskip-v4), EnvSpec(Berzerk-v0), EnvSpec(Berzerk-v4), EnvSpec(BerzerkDeterministic-v0), EnvSpec(BerzerkDeterministic-v4), EnvSpec(BerzerkNoFrameskip-v0), EnvSpec(BerzerkNoFrameskip-v4), EnvSpec(Berzerk-ram-v0), EnvSpec(Berzerk-ram-v4), EnvSpec(Berzerk-ramDeterministic-v0), EnvSpec(Berzerk-ramDeterministic-v4), EnvSpec(Berzerk-ramNoFrameskip-v0), EnvSpec(Berzerk-ramNoFrameskip-v4), EnvSpec(Bowling-v0), EnvSpec(Bowling-v4), EnvSpec(BowlingDeterministic-v0), EnvSpec(BowlingDeterministic-v4), EnvSpec(BowlingNoFrameskip-v0), EnvSpec(BowlingNoFrameskip-v4), EnvSpec(Bowling-ram-v0), EnvSpec(Bowling-ram-v4), EnvSpec(Bowling-ramDeterministic-v0), EnvSpec(Bowling-ramDeterministic-v4), EnvSpec(Bowling-ramNoFrameskip-v0), EnvSpec(Bowling-ramNoFrameskip-v4), EnvSpec(Boxing-v0), EnvSpec(Boxing-v4), EnvSpec(BoxingDeterministic-v0), EnvSpec(BoxingDeterministic-v4), EnvSpec(BoxingNoFrameskip-v0), EnvSpec(BoxingNoFrameskip-v4), EnvSpec(Boxing-ram-v0), EnvSpec(Boxing-ram-v4), EnvSpec(Boxing-ramDeterministic-v0), EnvSpec(Boxing-ramDeterministic-v4), EnvSpec(Boxing-ramNoFrameskip-v0), EnvSpec(Boxing-ramNoFrameskip-v4), EnvSpec(Breakout-v0), EnvSpec(Breakout-v4), EnvSpec(BreakoutDeterministic-v0), EnvSpec(BreakoutDeterministic-v4), EnvSpec(BreakoutNoFrameskip-v0), EnvSpec(BreakoutNoFrameskip-v4), EnvSpec(Breakout-ram-v0), EnvSpec(Breakout-ram-v4), EnvSpec(Breakout-ramDeterministic-v0), EnvSpec(Breakout-ramDeterministic-v4), EnvSpec(Breakout-ramNoFrameskip-v0), EnvSpec(Breakout-ramNoFrameskip-v4), EnvSpec(Carnival-v0), EnvSpec(Carnival-v4), EnvSpec(CarnivalDeterministic-v0), EnvSpec(CarnivalDeterministic-v4), EnvSpec(CarnivalNoFrameskip-v0), EnvSpec(CarnivalNoFrameskip-v4), EnvSpec(Carnival-ram-v0), EnvSpec(Carnival-ram-v4), EnvSpec(Carnival-ramDeterministic-v0), EnvSpec(Carnival-ramDeterministic-v4), EnvSpec(Carnival-ramNoFrameskip-v0), EnvSpec(Carnival-ramNoFrameskip-v4), EnvSpec(Centipede-v0), EnvSpec(Centipede-v4), EnvSpec(CentipedeDeterministic-v0), EnvSpec(CentipedeDeterministic-v4), EnvSpec(CentipedeNoFrameskip-v0), EnvSpec(CentipedeNoFrameskip-v4), EnvSpec(Centipede-ram-v0), EnvSpec(Centipede-ram-v4), EnvSpec(Centipede-ramDeterministic-v0), EnvSpec(Centipede-ramDeterministic-v4), EnvSpec(Centipede-ramNoFrameskip-v0), EnvSpec(Centipede-ramNoFrameskip-v4), EnvSpec(ChopperCommand-v0), EnvSpec(ChopperCommand-v4), EnvSpec(ChopperCommandDeterministic-v0), EnvSpec(ChopperCommandDeterministic-v4), EnvSpec(ChopperCommandNoFrameskip-v0), EnvSpec(ChopperCommandNoFrameskip-v4), EnvSpec(ChopperCommand-ram-v0), EnvSpec(ChopperCommand-ram-v4), EnvSpec(ChopperCommand-ramDeterministic-v0), EnvSpec(ChopperCommand-ramDeterministic-v4), EnvSpec(ChopperCommand-ramNoFrameskip-v0), EnvSpec(ChopperCommand-ramNoFrameskip-v4), EnvSpec(CrazyClimber-v0), EnvSpec(CrazyClimber-v4), EnvSpec(CrazyClimberDeterministic-v0), EnvSpec(CrazyClimberDeterministic-v4), EnvSpec(CrazyClimberNoFrameskip-v0), EnvSpec(CrazyClimberNoFrameskip-v4), EnvSpec(CrazyClimber-ram-v0), EnvSpec(CrazyClimber-ram-v4), EnvSpec(CrazyClimber-ramDeterministic-v0), EnvSpec(CrazyClimber-ramDeterministic-v4), EnvSpec(CrazyClimber-ramNoFrameskip-v0), EnvSpec(CrazyClimber-ramNoFrameskip-v4), EnvSpec(DemonAttack-v0), EnvSpec(DemonAttack-v4), EnvSpec(DemonAttackDeterministic-v0), EnvSpec(DemonAttackDeterministic-v4), EnvSpec(DemonAttackNoFrameskip-v0), EnvSpec(DemonAttackNoFrameskip-v4), EnvSpec(DemonAttack-ram-v0), EnvSpec(DemonAttack-ram-v4), EnvSpec(DemonAttack-ramDeterministic-v0), EnvSpec(DemonAttack-ramDeterministic-v4), EnvSpec(DemonAttack-ramNoFrameskip-v0), EnvSpec(DemonAttack-ramNoFrameskip-v4), EnvSpec(DoubleDunk-v0), EnvSpec(DoubleDunk-v4), EnvSpec(DoubleDunkDeterministic-v0), EnvSpec(DoubleDunkDeterministic-v4), EnvSpec(DoubleDunkNoFrameskip-v0), EnvSpec(DoubleDunkNoFrameskip-v4), EnvSpec(DoubleDunk-ram-v0), EnvSpec(DoubleDunk-ram-v4), EnvSpec(DoubleDunk-ramDeterministic-v0), EnvSpec(DoubleDunk-ramDeterministic-v4), EnvSpec(DoubleDunk-ramNoFrameskip-v0), EnvSpec(DoubleDunk-ramNoFrameskip-v4), EnvSpec(ElevatorAction-v0), EnvSpec(ElevatorAction-v4), EnvSpec(ElevatorActionDeterministic-v0), EnvSpec(ElevatorActionDeterministic-v4), EnvSpec(ElevatorActionNoFrameskip-v0), EnvSpec(ElevatorActionNoFrameskip-v4), EnvSpec(ElevatorAction-ram-v0), EnvSpec(ElevatorAction-ram-v4), EnvSpec(ElevatorAction-ramDeterministic-v0), EnvSpec(ElevatorAction-ramDeterministic-v4), EnvSpec(ElevatorAction-ramNoFrameskip-v0), EnvSpec(ElevatorAction-ramNoFrameskip-v4), EnvSpec(Enduro-v0), EnvSpec(Enduro-v4), EnvSpec(EnduroDeterministic-v0), EnvSpec(EnduroDeterministic-v4), EnvSpec(EnduroNoFrameskip-v0), EnvSpec(EnduroNoFrameskip-v4), EnvSpec(Enduro-ram-v0), EnvSpec(Enduro-ram-v4), EnvSpec(Enduro-ramDeterministic-v0), EnvSpec(Enduro-ramDeterministic-v4), EnvSpec(Enduro-ramNoFrameskip-v0), EnvSpec(Enduro-ramNoFrameskip-v4), EnvSpec(FishingDerby-v0), EnvSpec(FishingDerby-v4), EnvSpec(FishingDerbyDeterministic-v0), EnvSpec(FishingDerbyDeterministic-v4), EnvSpec(FishingDerbyNoFrameskip-v0), EnvSpec(FishingDerbyNoFrameskip-v4), EnvSpec(FishingDerby-ram-v0), EnvSpec(FishingDerby-ram-v4), EnvSpec(FishingDerby-ramDeterministic-v0), EnvSpec(FishingDerby-ramDeterministic-v4), EnvSpec(FishingDerby-ramNoFrameskip-v0), EnvSpec(FishingDerby-ramNoFrameskip-v4), EnvSpec(Freeway-v0), EnvSpec(Freeway-v4), EnvSpec(FreewayDeterministic-v0), EnvSpec(FreewayDeterministic-v4), EnvSpec(FreewayNoFrameskip-v0), EnvSpec(FreewayNoFrameskip-v4), EnvSpec(Freeway-ram-v0), EnvSpec(Freeway-ram-v4), EnvSpec(Freeway-ramDeterministic-v0), EnvSpec(Freeway-ramDeterministic-v4), EnvSpec(Freeway-ramNoFrameskip-v0), EnvSpec(Freeway-ramNoFrameskip-v4), EnvSpec(Frostbite-v0), EnvSpec(Frostbite-v4), EnvSpec(FrostbiteDeterministic-v0), EnvSpec(FrostbiteDeterministic-v4), EnvSpec(FrostbiteNoFrameskip-v0), EnvSpec(FrostbiteNoFrameskip-v4), EnvSpec(Frostbite-ram-v0), EnvSpec(Frostbite-ram-v4), EnvSpec(Frostbite-ramDeterministic-v0), EnvSpec(Frostbite-ramDeterministic-v4), EnvSpec(Frostbite-ramNoFrameskip-v0), EnvSpec(Frostbite-ramNoFrameskip-v4), EnvSpec(Gopher-v0), EnvSpec(Gopher-v4), EnvSpec(GopherDeterministic-v0), EnvSpec(GopherDeterministic-v4), EnvSpec(GopherNoFrameskip-v0), EnvSpec(GopherNoFrameskip-v4), EnvSpec(Gopher-ram-v0), EnvSpec(Gopher-ram-v4), EnvSpec(Gopher-ramDeterministic-v0), EnvSpec(Gopher-ramDeterministic-v4), EnvSpec(Gopher-ramNoFrameskip-v0), EnvSpec(Gopher-ramNoFrameskip-v4), EnvSpec(Gravitar-v0), EnvSpec(Gravitar-v4), EnvSpec(GravitarDeterministic-v0), EnvSpec(GravitarDeterministic-v4), EnvSpec(GravitarNoFrameskip-v0), EnvSpec(GravitarNoFrameskip-v4), EnvSpec(Gravitar-ram-v0), EnvSpec(Gravitar-ram-v4), EnvSpec(Gravitar-ramDeterministic-v0), EnvSpec(Gravitar-ramDeterministic-v4), EnvSpec(Gravitar-ramNoFrameskip-v0), EnvSpec(Gravitar-ramNoFrameskip-v4), EnvSpec(Hero-v0), EnvSpec(Hero-v4), EnvSpec(HeroDeterministic-v0), EnvSpec(HeroDeterministic-v4), EnvSpec(HeroNoFrameskip-v0), EnvSpec(HeroNoFrameskip-v4), EnvSpec(Hero-ram-v0), EnvSpec(Hero-ram-v4), EnvSpec(Hero-ramDeterministic-v0), EnvSpec(Hero-ramDeterministic-v4), EnvSpec(Hero-ramNoFrameskip-v0), EnvSpec(Hero-ramNoFrameskip-v4), EnvSpec(IceHockey-v0), EnvSpec(IceHockey-v4), EnvSpec(IceHockeyDeterministic-v0), EnvSpec(IceHockeyDeterministic-v4), EnvSpec(IceHockeyNoFrameskip-v0), EnvSpec(IceHockeyNoFrameskip-v4), EnvSpec(IceHockey-ram-v0), EnvSpec(IceHockey-ram-v4), EnvSpec(IceHockey-ramDeterministic-v0), EnvSpec(IceHockey-ramDeterministic-v4), EnvSpec(IceHockey-ramNoFrameskip-v0), EnvSpec(IceHockey-ramNoFrameskip-v4), EnvSpec(Jamesbond-v0), EnvSpec(Jamesbond-v4), EnvSpec(JamesbondDeterministic-v0), EnvSpec(JamesbondDeterministic-v4), EnvSpec(JamesbondNoFrameskip-v0), EnvSpec(JamesbondNoFrameskip-v4), EnvSpec(Jamesbond-ram-v0), EnvSpec(Jamesbond-ram-v4), EnvSpec(Jamesbond-ramDeterministic-v0), EnvSpec(Jamesbond-ramDeterministic-v4), EnvSpec(Jamesbond-ramNoFrameskip-v0), EnvSpec(Jamesbond-ramNoFrameskip-v4), EnvSpec(JourneyEscape-v0), EnvSpec(JourneyEscape-v4), EnvSpec(JourneyEscapeDeterministic-v0), EnvSpec(JourneyEscapeDeterministic-v4), EnvSpec(JourneyEscapeNoFrameskip-v0), EnvSpec(JourneyEscapeNoFrameskip-v4), EnvSpec(JourneyEscape-ram-v0), EnvSpec(JourneyEscape-ram-v4), EnvSpec(JourneyEscape-ramDeterministic-v0), EnvSpec(JourneyEscape-ramDeterministic-v4), EnvSpec(JourneyEscape-ramNoFrameskip-v0), EnvSpec(JourneyEscape-ramNoFrameskip-v4), EnvSpec(Kangaroo-v0), EnvSpec(Kangaroo-v4), EnvSpec(KangarooDeterministic-v0), EnvSpec(KangarooDeterministic-v4), EnvSpec(KangarooNoFrameskip-v0), EnvSpec(KangarooNoFrameskip-v4), EnvSpec(Kangaroo-ram-v0), EnvSpec(Kangaroo-ram-v4), EnvSpec(Kangaroo-ramDeterministic-v0), EnvSpec(Kangaroo-ramDeterministic-v4), EnvSpec(Kangaroo-ramNoFrameskip-v0), EnvSpec(Kangaroo-ramNoFrameskip-v4), EnvSpec(Krull-v0), EnvSpec(Krull-v4), EnvSpec(KrullDeterministic-v0), EnvSpec(KrullDeterministic-v4), EnvSpec(KrullNoFrameskip-v0), EnvSpec(KrullNoFrameskip-v4), EnvSpec(Krull-ram-v0), EnvSpec(Krull-ram-v4), EnvSpec(Krull-ramDeterministic-v0), EnvSpec(Krull-ramDeterministic-v4), EnvSpec(Krull-ramNoFrameskip-v0), EnvSpec(Krull-ramNoFrameskip-v4), EnvSpec(KungFuMaster-v0), EnvSpec(KungFuMaster-v4), EnvSpec(KungFuMasterDeterministic-v0), EnvSpec(KungFuMasterDeterministic-v4), EnvSpec(KungFuMasterNoFrameskip-v0), EnvSpec(KungFuMasterNoFrameskip-v4), EnvSpec(KungFuMaster-ram-v0), EnvSpec(KungFuMaster-ram-v4), EnvSpec(KungFuMaster-ramDeterministic-v0), EnvSpec(KungFuMaster-ramDeterministic-v4), EnvSpec(KungFuMaster-ramNoFrameskip-v0), EnvSpec(KungFuMaster-ramNoFrameskip-v4), EnvSpec(MontezumaRevenge-v0), EnvSpec(MontezumaRevenge-v4), EnvSpec(MontezumaRevengeDeterministic-v0), EnvSpec(MontezumaRevengeDeterministic-v4), EnvSpec(MontezumaRevengeNoFrameskip-v0), EnvSpec(MontezumaRevengeNoFrameskip-v4), EnvSpec(MontezumaRevenge-ram-v0), EnvSpec(MontezumaRevenge-ram-v4), EnvSpec(MontezumaRevenge-ramDeterministic-v0), EnvSpec(MontezumaRevenge-ramDeterministic-v4), EnvSpec(MontezumaRevenge-ramNoFrameskip-v0), EnvSpec(MontezumaRevenge-ramNoFrameskip-v4), EnvSpec(MsPacman-v0), EnvSpec(MsPacman-v4), EnvSpec(MsPacmanDeterministic-v0), EnvSpec(MsPacmanDeterministic-v4), EnvSpec(MsPacmanNoFrameskip-v0), EnvSpec(MsPacmanNoFrameskip-v4), EnvSpec(MsPacman-ram-v0), EnvSpec(MsPacman-ram-v4), EnvSpec(MsPacman-ramDeterministic-v0), EnvSpec(MsPacman-ramDeterministic-v4), EnvSpec(MsPacman-ramNoFrameskip-v0), EnvSpec(MsPacman-ramNoFrameskip-v4), EnvSpec(NameThisGame-v0), EnvSpec(NameThisGame-v4), EnvSpec(NameThisGameDeterministic-v0), EnvSpec(NameThisGameDeterministic-v4), EnvSpec(NameThisGameNoFrameskip-v0), EnvSpec(NameThisGameNoFrameskip-v4), EnvSpec(NameThisGame-ram-v0), EnvSpec(NameThisGame-ram-v4), EnvSpec(NameThisGame-ramDeterministic-v0), EnvSpec(NameThisGame-ramDeterministic-v4), EnvSpec(NameThisGame-ramNoFrameskip-v0), EnvSpec(NameThisGame-ramNoFrameskip-v4), EnvSpec(Phoenix-v0), EnvSpec(Phoenix-v4), EnvSpec(PhoenixDeterministic-v0), EnvSpec(PhoenixDeterministic-v4), EnvSpec(PhoenixNoFrameskip-v0), EnvSpec(PhoenixNoFrameskip-v4), EnvSpec(Phoenix-ram-v0), EnvSpec(Phoenix-ram-v4), EnvSpec(Phoenix-ramDeterministic-v0), EnvSpec(Phoenix-ramDeterministic-v4), EnvSpec(Phoenix-ramNoFrameskip-v0), EnvSpec(Phoenix-ramNoFrameskip-v4), EnvSpec(Pitfall-v0), EnvSpec(Pitfall-v4), EnvSpec(PitfallDeterministic-v0), EnvSpec(PitfallDeterministic-v4), EnvSpec(PitfallNoFrameskip-v0), EnvSpec(PitfallNoFrameskip-v4), EnvSpec(Pitfall-ram-v0), EnvSpec(Pitfall-ram-v4), EnvSpec(Pitfall-ramDeterministic-v0), EnvSpec(Pitfall-ramDeterministic-v4), EnvSpec(Pitfall-ramNoFrameskip-v0), EnvSpec(Pitfall-ramNoFrameskip-v4), EnvSpec(Pong-v0), EnvSpec(Pong-v4), EnvSpec(PongDeterministic-v0), EnvSpec(PongDeterministic-v4), EnvSpec(PongNoFrameskip-v0), EnvSpec(PongNoFrameskip-v4), EnvSpec(Pong-ram-v0), EnvSpec(Pong-ram-v4), EnvSpec(Pong-ramDeterministic-v0), EnvSpec(Pong-ramDeterministic-v4), EnvSpec(Pong-ramNoFrameskip-v0), EnvSpec(Pong-ramNoFrameskip-v4), EnvSpec(Pooyan-v0), EnvSpec(Pooyan-v4), EnvSpec(PooyanDeterministic-v0), EnvSpec(PooyanDeterministic-v4), EnvSpec(PooyanNoFrameskip-v0), EnvSpec(PooyanNoFrameskip-v4), EnvSpec(Pooyan-ram-v0), EnvSpec(Pooyan-ram-v4), EnvSpec(Pooyan-ramDeterministic-v0), EnvSpec(Pooyan-ramDeterministic-v4), EnvSpec(Pooyan-ramNoFrameskip-v0), EnvSpec(Pooyan-ramNoFrameskip-v4), EnvSpec(PrivateEye-v0), EnvSpec(PrivateEye-v4), EnvSpec(PrivateEyeDeterministic-v0), EnvSpec(PrivateEyeDeterministic-v4), EnvSpec(PrivateEyeNoFrameskip-v0), EnvSpec(PrivateEyeNoFrameskip-v4), EnvSpec(PrivateEye-ram-v0), EnvSpec(PrivateEye-ram-v4), EnvSpec(PrivateEye-ramDeterministic-v0), EnvSpec(PrivateEye-ramDeterministic-v4), EnvSpec(PrivateEye-ramNoFrameskip-v0), EnvSpec(PrivateEye-ramNoFrameskip-v4), EnvSpec(Qbert-v0), EnvSpec(Qbert-v4), EnvSpec(QbertDeterministic-v0), EnvSpec(QbertDeterministic-v4), EnvSpec(QbertNoFrameskip-v0), EnvSpec(QbertNoFrameskip-v4), EnvSpec(Qbert-ram-v0), EnvSpec(Qbert-ram-v4), EnvSpec(Qbert-ramDeterministic-v0), EnvSpec(Qbert-ramDeterministic-v4), EnvSpec(Qbert-ramNoFrameskip-v0), EnvSpec(Qbert-ramNoFrameskip-v4), EnvSpec(Riverraid-v0), EnvSpec(Riverraid-v4), EnvSpec(RiverraidDeterministic-v0), EnvSpec(RiverraidDeterministic-v4), EnvSpec(RiverraidNoFrameskip-v0), EnvSpec(RiverraidNoFrameskip-v4), EnvSpec(Riverraid-ram-v0), EnvSpec(Riverraid-ram-v4), EnvSpec(Riverraid-ramDeterministic-v0), EnvSpec(Riverraid-ramDeterministic-v4), EnvSpec(Riverraid-ramNoFrameskip-v0), EnvSpec(Riverraid-ramNoFrameskip-v4), EnvSpec(RoadRunner-v0), EnvSpec(RoadRunner-v4), EnvSpec(RoadRunnerDeterministic-v0), EnvSpec(RoadRunnerDeterministic-v4), EnvSpec(RoadRunnerNoFrameskip-v0), EnvSpec(RoadRunnerNoFrameskip-v4), EnvSpec(RoadRunner-ram-v0), EnvSpec(RoadRunner-ram-v4), EnvSpec(RoadRunner-ramDeterministic-v0), EnvSpec(RoadRunner-ramDeterministic-v4), EnvSpec(RoadRunner-ramNoFrameskip-v0), EnvSpec(RoadRunner-ramNoFrameskip-v4), EnvSpec(Robotank-v0), EnvSpec(Robotank-v4), EnvSpec(RobotankDeterministic-v0), EnvSpec(RobotankDeterministic-v4), EnvSpec(RobotankNoFrameskip-v0), EnvSpec(RobotankNoFrameskip-v4), EnvSpec(Robotank-ram-v0), EnvSpec(Robotank-ram-v4), EnvSpec(Robotank-ramDeterministic-v0), EnvSpec(Robotank-ramDeterministic-v4), EnvSpec(Robotank-ramNoFrameskip-v0), EnvSpec(Robotank-ramNoFrameskip-v4), EnvSpec(Seaquest-v0), EnvSpec(Seaquest-v4), EnvSpec(SeaquestDeterministic-v0), EnvSpec(SeaquestDeterministic-v4), EnvSpec(SeaquestNoFrameskip-v0), EnvSpec(SeaquestNoFrameskip-v4), EnvSpec(Seaquest-ram-v0), EnvSpec(Seaquest-ram-v4), EnvSpec(Seaquest-ramDeterministic-v0), EnvSpec(Seaquest-ramDeterministic-v4), EnvSpec(Seaquest-ramNoFrameskip-v0), EnvSpec(Seaquest-ramNoFrameskip-v4), EnvSpec(Skiing-v0), EnvSpec(Skiing-v4), EnvSpec(SkiingDeterministic-v0), EnvSpec(SkiingDeterministic-v4), EnvSpec(SkiingNoFrameskip-v0), EnvSpec(SkiingNoFrameskip-v4), EnvSpec(Skiing-ram-v0), EnvSpec(Skiing-ram-v4), EnvSpec(Skiing-ramDeterministic-v0), EnvSpec(Skiing-ramDeterministic-v4), EnvSpec(Skiing-ramNoFrameskip-v0), EnvSpec(Skiing-ramNoFrameskip-v4), EnvSpec(Solaris-v0), EnvSpec(Solaris-v4), EnvSpec(SolarisDeterministic-v0), EnvSpec(SolarisDeterministic-v4), EnvSpec(SolarisNoFrameskip-v0), EnvSpec(SolarisNoFrameskip-v4), EnvSpec(Solaris-ram-v0), EnvSpec(Solaris-ram-v4), EnvSpec(Solaris-ramDeterministic-v0), EnvSpec(Solaris-ramDeterministic-v4), EnvSpec(Solaris-ramNoFrameskip-v0), EnvSpec(Solaris-ramNoFrameskip-v4), EnvSpec(SpaceInvaders-v0), EnvSpec(SpaceInvaders-v4), EnvSpec(SpaceInvadersDeterministic-v0), EnvSpec(SpaceInvadersDeterministic-v4), EnvSpec(SpaceInvadersNoFrameskip-v0), EnvSpec(SpaceInvadersNoFrameskip-v4), EnvSpec(SpaceInvaders-ram-v0), EnvSpec(SpaceInvaders-ram-v4), EnvSpec(SpaceInvaders-ramDeterministic-v0), EnvSpec(SpaceInvaders-ramDeterministic-v4), EnvSpec(SpaceInvaders-ramNoFrameskip-v0), EnvSpec(SpaceInvaders-ramNoFrameskip-v4), EnvSpec(StarGunner-v0), EnvSpec(StarGunner-v4), EnvSpec(StarGunnerDeterministic-v0), EnvSpec(StarGunnerDeterministic-v4), EnvSpec(StarGunnerNoFrameskip-v0), EnvSpec(StarGunnerNoFrameskip-v4), EnvSpec(StarGunner-ram-v0), EnvSpec(StarGunner-ram-v4), EnvSpec(StarGunner-ramDeterministic-v0), EnvSpec(StarGunner-ramDeterministic-v4), EnvSpec(StarGunner-ramNoFrameskip-v0), EnvSpec(StarGunner-ramNoFrameskip-v4), EnvSpec(Tennis-v0), EnvSpec(Tennis-v4), EnvSpec(TennisDeterministic-v0), EnvSpec(TennisDeterministic-v4), EnvSpec(TennisNoFrameskip-v0), EnvSpec(TennisNoFrameskip-v4), EnvSpec(Tennis-ram-v0), EnvSpec(Tennis-ram-v4), EnvSpec(Tennis-ramDeterministic-v0), EnvSpec(Tennis-ramDeterministic-v4), EnvSpec(Tennis-ramNoFrameskip-v0), EnvSpec(Tennis-ramNoFrameskip-v4), EnvSpec(TimePilot-v0), EnvSpec(TimePilot-v4), EnvSpec(TimePilotDeterministic-v0), EnvSpec(TimePilotDeterministic-v4), EnvSpec(TimePilotNoFrameskip-v0), EnvSpec(TimePilotNoFrameskip-v4), EnvSpec(TimePilot-ram-v0), EnvSpec(TimePilot-ram-v4), EnvSpec(TimePilot-ramDeterministic-v0), EnvSpec(TimePilot-ramDeterministic-v4), EnvSpec(TimePilot-ramNoFrameskip-v0), EnvSpec(TimePilot-ramNoFrameskip-v4), EnvSpec(Tutankham-v0), EnvSpec(Tutankham-v4), EnvSpec(TutankhamDeterministic-v0), EnvSpec(TutankhamDeterministic-v4), EnvSpec(TutankhamNoFrameskip-v0), EnvSpec(TutankhamNoFrameskip-v4), EnvSpec(Tutankham-ram-v0), EnvSpec(Tutankham-ram-v4), EnvSpec(Tutankham-ramDeterministic-v0), EnvSpec(Tutankham-ramDeterministic-v4), EnvSpec(Tutankham-ramNoFrameskip-v0), EnvSpec(Tutankham-ramNoFrameskip-v4), EnvSpec(UpNDown-v0), EnvSpec(UpNDown-v4), EnvSpec(UpNDownDeterministic-v0), EnvSpec(UpNDownDeterministic-v4), EnvSpec(UpNDownNoFrameskip-v0), EnvSpec(UpNDownNoFrameskip-v4), EnvSpec(UpNDown-ram-v0), EnvSpec(UpNDown-ram-v4), EnvSpec(UpNDown-ramDeterministic-v0), EnvSpec(UpNDown-ramDeterministic-v4), EnvSpec(UpNDown-ramNoFrameskip-v0), EnvSpec(UpNDown-ramNoFrameskip-v4), EnvSpec(Venture-v0), EnvSpec(Venture-v4), EnvSpec(VentureDeterministic-v0), EnvSpec(VentureDeterministic-v4), EnvSpec(VentureNoFrameskip-v0), EnvSpec(VentureNoFrameskip-v4), EnvSpec(Venture-ram-v0), EnvSpec(Venture-ram-v4), EnvSpec(Venture-ramDeterministic-v0), EnvSpec(Venture-ramDeterministic-v4), EnvSpec(Venture-ramNoFrameskip-v0), EnvSpec(Venture-ramNoFrameskip-v4), EnvSpec(VideoPinball-v0), EnvSpec(VideoPinball-v4), EnvSpec(VideoPinballDeterministic-v0), EnvSpec(VideoPinballDeterministic-v4), EnvSpec(VideoPinballNoFrameskip-v0), EnvSpec(VideoPinballNoFrameskip-v4), EnvSpec(VideoPinball-ram-v0), EnvSpec(VideoPinball-ram-v4), EnvSpec(VideoPinball-ramDeterministic-v0), EnvSpec(VideoPinball-ramDeterministic-v4), EnvSpec(VideoPinball-ramNoFrameskip-v0), EnvSpec(VideoPinball-ramNoFrameskip-v4), EnvSpec(WizardOfWor-v0), EnvSpec(WizardOfWor-v4), EnvSpec(WizardOfWorDeterministic-v0), EnvSpec(WizardOfWorDeterministic-v4), EnvSpec(WizardOfWorNoFrameskip-v0), EnvSpec(WizardOfWorNoFrameskip-v4), EnvSpec(WizardOfWor-ram-v0), EnvSpec(WizardOfWor-ram-v4), EnvSpec(WizardOfWor-ramDeterministic-v0), EnvSpec(WizardOfWor-ramDeterministic-v4), EnvSpec(WizardOfWor-ramNoFrameskip-v0), EnvSpec(WizardOfWor-ramNoFrameskip-v4), EnvSpec(YarsRevenge-v0), EnvSpec(YarsRevenge-v4), EnvSpec(YarsRevengeDeterministic-v0), EnvSpec(YarsRevengeDeterministic-v4), EnvSpec(YarsRevengeNoFrameskip-v0), EnvSpec(YarsRevengeNoFrameskip-v4), EnvSpec(YarsRevenge-ram-v0), EnvSpec(YarsRevenge-ram-v4), EnvSpec(YarsRevenge-ramDeterministic-v0), EnvSpec(YarsRevenge-ramDeterministic-v4), EnvSpec(YarsRevenge-ramNoFrameskip-v0), EnvSpec(YarsRevenge-ramNoFrameskip-v4), EnvSpec(Zaxxon-v0), EnvSpec(Zaxxon-v4), EnvSpec(ZaxxonDeterministic-v0), EnvSpec(ZaxxonDeterministic-v4), EnvSpec(ZaxxonNoFrameskip-v0), EnvSpec(ZaxxonNoFrameskip-v4), EnvSpec(Zaxxon-ram-v0), EnvSpec(Zaxxon-ram-v4), EnvSpec(Zaxxon-ramDeterministic-v0), EnvSpec(Zaxxon-ramDeterministic-v4), EnvSpec(Zaxxon-ramNoFrameskip-v0), EnvSpec(Zaxxon-ramNoFrameskip-v4), EnvSpec(CubeCrash-v0), EnvSpec(CubeCrashSparse-v0), EnvSpec(CubeCrashScreenBecomesBlack-v0), EnvSpec(MemorizeDigits-v0)])
```
You could also add your own environments using `register()`. 

## Set up OpenAI Gym

### Installation
If you are using an IDE, activate the virtual environment you've been using (`activate venv` for windows users and `source activate venv` for Linux and macOS users) and pip install gym:
```
pip install gym
```
**Warning**: To install the gym library, you will need to have Python3.5 or above. 

If you wish to continue using Google Colab, first install the dependencies: 
```
!apt-get install -y xvfb python-opengl > /dev/null 2>&1
```
Then install the libraries required:
```
!pip -q install gym
!pip -q install pyglet
!pip -q install pyopengl
!pip -q install pyvirtualdisplay
```
## Cart-pole balancing 

### Task
A pole is attached by an un-actuated joint to a cart which moves along a frictionless track.  The task is to apply forces to the cart along the track, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

### Action
Two discrete actions: a=0 indicates pushing cart to the left and a=1 pushes the cart to the right.  The amount the velocity reduced or increased does not only depends on the direction you are moving but also on the angle the pole is pointing. 

### Reward
 Reward is +1 for every step taken, including the termination step.

### Observation
There are 4 observations returned by the environment after each action taken by an agent:
- Cart position:  a number between `-4.8` and `4.8`
- Cart velocity: a number between `-inf`and `inf`
- Pole angle: an angle between -24&deg; and 24&deg; 
-  Pole velocity at tip: a number between `-inf`and `inf`

### Termination
- Cart position is smaller or greater than `-2.4` or `2.4`
- Pole Angle is smaller or greater than -12&deg; or 12&deg;
- Episode length is longer than 200

### Solved requirements
The problem is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

## Basics in Gym

### Import an environment
To start with, let's choose the cart-pole environment. The following command returns the environment passed as argument.
```
gym.make('environment_id')
```
Every environment comes with an `action_space` and an `observation_space`. For example, 
```
import gym

env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
```
```
>>>Discrete(2) 
   Box(4,)
```
Action space shows the number of actions one can take and observation space. In this case, the actions include `left` and `right`. `Box(4, )` indicates that you have 4 state variables: cart position, cart velocity, pole angle and pole position.
You can also check the bounds of these two variables: 
```
print(env.observation_space.high)
print(env.observation_space.low)
```
```
>>>[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38] 
   [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
```
The angle boundary is calculated by $`24 * 2 * \pi /360`$.  When angle value lies outside `(-0.209, 0.209)` (computed by $`± 12 * 2 * \pi /360`$), you will fail the episode. 

### Start the process
Start the process by calling `env.reset()`.  This function creates a new environment and returns an initial observation. 

`env.render()` displays a popup window that renders an update for each step in the loop. 

### The gym `Step` function
The environment's `step` function takes an action at each time step.  The action is passed as an argument to the function.  `step` function returns four values, namely `observation`, `reward`, `done` and `info`. 

`observation`:  an environment-specific object representing your observation of the environment.

`reward`: amount of reward achieved by the previous action.

`done`: a boolean value. When it's true, it indicates the episode has terminated and the environment needs to be reset. 

`info`:  diagnostic information useful for debugging.

### Taking a random action
`env.action_space.sample()` takes a random action. 

## Get something running
Let's try to run the `CartPole-v0` environment for 100 timesteps taking random actions. 

If you are using an IDE, the following code will work:
```
import gym  
  
env = gym.make("CartPole-v0")  
env.reset()  
  
for _ in range(100):  
    env.render()  
    env.step(env.action_space.sample())  
  
env.close()
```
You should see a window pop up rendering the Cart-pole problem. 

![Cart-pole problem](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/Cart-Pole.mp4)

It's a little bit more complicated in Google Colab as you will have to record the frames of the games and render the display through HTML.

Firstly, import the dependencies, 
```
!apt-get install -y xvfb python-opengl > /dev/null 2>&1
```

Import all the libraries needed, 
```
import gym
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
```
Then, to activate virtual display we need to run the following code once for training an agent:
```
# Start virtual display
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()
```

Record the frames and display as an animation,
```
env = gym.make('CartPole-v0')
env.reset()

frames = []
for i in range(100):
    frames.append(env.render(mode = 'rgb_array'))
    env.step(env.action_space.sample())

plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
patch = plt.imshow(frames[0])
plt.axis('off')
animate = lambda i: patch.set_data(frames[i])
ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
HTML(ani.to_jshtml())
```

You should be able to see an animation in the output cell:

![alt text](https://github.com/lse-st449/lectures/raw/master/Week07/class/graphs/cart_pole_colab.png)

### Save the video
If you are using an IDE, simply add the following code before calling reset
```
env = gym.wrappers.Monitor(env, './', force=True)
```

In Google Colab,  you will have to install `ffmpeg` using 
```
!apt-get install ffmpeg
```
Then run the same code as in an IDE, 
```
import gym
env = gym.make("CartPoler-v0")
env = gym.wrappers.Monitor(env, "./", force=True)
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())

env.close()
```
Use Python `glob` module to find the pathnames matching a specified pattern and download it,
```
from google.colab import files
import glob

for file in glob.glob("openaigym.video.*.mp4"):
  files.download(file)
```

### Run episodes
Run 10 episodes with random actions,
```
episodes = 10
for i in range(episodes):
  env.reset()
  done = False
  rewards = 0
  while not done and rewards < 200:
    frames.append(env.render(mode = 'rgb_array'))
    obs, r, done, info = env.step(env.action_space.sample())
    rewards += r
  print('Episode:', i, 'Rewards:', rewards)
  print('Observations:', obs)

```

```
>>>Episode: 0 Rewards: 12.0 
Observations: [ 0.09807745 1.13328445 -0.23175972 -2.05475785] 
>Episode: 1 Rewards: 12.0 
Observations: [-0.16441889 -0.79095032 0.22053015 1.475422 ] 
>Episode: 2 Rewards: 16.0 
Observations: [ 0.10420795 0.74351681 -0.20948748 -1.43308032]
>Episode: 3 Rewards: 14.0 
Observations: [ 0.14296193 1.15201322 -0.23373265 -2.12840362] 
>Episode: 4 Rewards: 27.0 
Observations: [-0.14729198 -0.96910426 0.23477009 1.71584913] 
>Episode: 5 Rewards: 12.0 
Observations: [ 0.18810051 1.1513385 -0.25134098 -2.00758437] 
>Episode: 6 Rewards: 23.0 
Observations: [-0.19522666 -1.41476826 0.23308203 2.29550233] 
>Episode: 7 Rewards: 13.0 
Observations: [ 0.13458203 1.32632518 -0.21941213 -2.27169373] 
>Episode: 8 Rewards: 24.0 
Observations: [-0.07062128 -0.42954568 0.22251542 1.22227352] 
>Episode: 9 Rewards: 21.0 
Observations: [ 0.18714034 0.25414172 -0.21798099 -0.76434372]
```
The third variable in the observations corresponds to angles which seemed to be the reason why the episodes failed. Therefore, we design a naive improvement on the previous algorithm. Instead of choosing random actions.  we take actions based on the pole angles, i.e., when the angle is negative, we push the cart to the left, otherwise to the right. 
```
def action(angle):
    if angle < 0:
        return 0
    else:
        return 1
```
And modify the execution part:
```
episodes = 10
for i in range(episodes):
  env.reset()
  done = False
  obs, r, _, _ = env.step(env.action_space.sample())
  rewards = r
  while not done and rewards < 200:
    frames.append(env.render(mode = 'rgb_array'))
    a = action(obs[2])
    obs, r, done, info = env.step(a)
    rewards += r
  print('Episode:', i, 'Rewards:', rewards)
  print('Observations:', obs)
```
```
>>>Episode: 0 Rewards: 41.0 
Observations: [-0.30933015 -0.57799895 0.21229856 0.53278391] 
>Episode: 1 Rewards: 46.0 
Observations: [-0.21721116 -0.38197746 0.21531054 0.48483688] 
>Episode: 2 Rewards: 56.0 
Observations: [ 0.18672662 0.71894957 -0.21140071 -0.84340343] 
>Episode: 3 Rewards: 35.0 
Observations: [ 0.15831088 0.19682541 -0.21454678 -0.33491341] 
>Episode: 4 Rewards: 28.0 
Observations: [-0.12039557 -0.81733381 0.21721926 1.05910844] 
>Episode: 5 Rewards: 54.0 
Observations: [ 0.17445683 1.17851513 -0.21097441 -1.51606576] 
>Episode: 6 Rewards: 43.0 
Observations: [ 0.27277641 0.537695 -0.22499787 -0.58432232] 
>Episode: 7 Rewards: 54.0 
Observations: [ 0.1617538 1.19422305 -0.21623197 -1.58552206] 
>Episode: 8 Rewards: 40.0 
Observations: [ 0.16447311 0.71950543 -0.22039137 -0.86368136] 
>Episode: 9 Rewards: 37.0 
Observations: [-0.21866421 -0.91799244 0.23148745 1.17893848]
```

## Homework: multi-armed bandits

As discussed in Lecture 6, the k-armed bandit problem is a classic learning problem that well demonstrates the dilemma between exploration and exploitation when making decisions.

In this exercise, our goal is to implement and evaluate the performance of the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="a"/>-greedy algorithm for the four Bernoulli arms example covered in the lecture.

Recall that the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="a"/>-greedy algorithm at each decision time, makes a greedy choice with probability <img src="https://latex.codecogs.com/svg.latex?\Large&space;1-\epsilon" title="a"/>, and, otherwise, makes an explorative step by selecting an arm uniformly at random from the set of all arms. We have discussed in the lecture how adding such an explorative step can outperform a greedy algorithm (that always makes greedy choices) with respect to different performance metrics. In particular, we considered the regret defined as follows  

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathrm{Regret}(T)=Q^*(a^*)T-\sum_{t=1}^TQ^*(a_t)" title="a"/>

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(a)" title="a"/> is the expected reward of arm <img src="https://latex.codecogs.com/svg.latex?\Large&space;a" title="a"/>, <img src="https://latex.codecogs.com/svg.latex?\Large&space;a^*" title="a"/> is a best arm, and <img src="https://latex.codecogs.com/svg.latex?\Large&space;a_t" title="a"/> is the arm selected by the algorithm at time step <img src="https://latex.codecogs.com/svg.latex?\Large&space;t" title="a"/>.

The <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="a"/>-greedy algorithm has a regret that linearly increases with time <img src="https://latex.codecogs.com/svg.latex?\Large&space;T" title="a"/>. This is easy to observe as follows. Let <img src="https://latex.codecogs.com/svg.latex?\Large&space;d" title="a"/> be a positive parameter such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(a^*)-Q^*(a)\geqd" title="a"/> for all <img src="https://latex.codecogs.com/svg.latex?\Large&space;a" title="a"/> such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(a)<Q^*(a^*)" title="a"/>. We call any arm <img src="https://latex.codecogs.com/svg.latex?\Large&space;a" title="a"/> such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(a) < Q^*(a^*)" title="a"/> a non-best arm. Then, we can write

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathrm{Regret}(T)\geqdM_T" title="a"/>

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;M_T" title="a"/> is the number of times a non-best arm is selected by the algorithm. Now, for the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="a"/>-greedy algorithm, at each time step, a non-best arm is pulled with probability at least <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon/k" title="a"/>, which combined with the law of large numbers yields <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lim_{T\rightarrow\infty}M_T/T\geq\epsilon/k" title="a"/>. Therefore, it follows

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathrm{Regret}(T)=\Omega(T)" title="a"/> with high probability.

The question is whether there exists an algorithm that can guarantee a sublinear growth of regret with time <img src="https://latex.codecogs.com/svg.latex?\Large&space;T" title="a"/>. 

It is well known that for stochastic multi-armed bandit problems the optimal growth of the regret is logarithmic in <img src="https://latex.codecogs.com/svg.latex?\Large&space;T" title="a"/>. This optimal regret growth can be achieved by an <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon_t" title="a"/>-greedy algorithm for which the probability of making an explorative step <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon_t" title="a"/> eventually decreases with time <img src="https://latex.codecogs.com/svg.latex?\Large&space;t" title="a"/> in a specific way. Concretely, this holds by defining 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon_t = \max\left\{\frac{ck}{d^2 t},1\right\}" title="a"/>

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;c>0" title="a"/> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;0<d<1" title="a"/> are parameters.

For any <img src="https://latex.codecogs.com/svg.latex?\Large&space;k" title="a"/>-armed bandit problem such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(a)\in[0,1]" title="a"/> for each arm <img src="https://latex.codecogs.com/svg.latex?\Large&space;a" title="a"/> and such that <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(a^*)-Q^*(a)\geqd" title="a"/> for each non-best arm <img src="https://latex.codecogs.com/svg.latex?\Large&space;a" title="a"/>, the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon_t" title="a"/>-greedy algorithm with <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon_t" title="a"/> defined as above, has the regret bound:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathrm{Regret}(T)=O((k/d^2)\log(T))" title="a"/>.

This result follows from Theorem 3 in the paper [Auer, Cesa-Bianchi and Fischer (2002)](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf). 

The goal of this exercise is to empirically validate this theoretical result. 

- Environment setup 
  - Assume k = 4
  -  the reward of each arm follows a Bernoulli distribution
  - the Bernoulli parameters <img src="https://latex.codecogs.com/svg.latex?\Large&space;p=[0.1,0.4,0.1,0.1]" title="a" />  for all 4 arms
  - fix seed `np.random.seed(3)`

In order to keep consistency throughout the rest of the seminars, we will code the environment in the same manner as OpenAI gym, in which the step function take action as its only argument:
```
# environment
bandit_probs = [0.10, 0.40, 0.10, 0.10]

def step(action):
    rand = np.random.random()  # [0.0,1.0)
    reward = 1.0 if (rand < bandit_probs[action]) else 0.0
    return reward
```

**Tasks**: 
- implement the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="a"/>-greedy algorithm explained in the lecture
- compare the results for various <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon" title="a" /> values
  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon=[0.05, 0.1, 0.2]" title="a" /> 
- implement the <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon_t" title="a" />-greedy algorithm with <img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon_t" title="a"/> as defined above 
  - set <img src="https://latex.codecogs.com/svg.latex?\Large&space;c=0.1" title="a" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;d=0.3" title="a" />
- run a total number of 50 runs with each run consisting of 1000 time steps
- compute the average rewards and regrets at each time step for 50 trials
- plot the cumulative average rewards and regrets vs total number of time steps
