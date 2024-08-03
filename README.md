# MLB Pitch Classification with Neural Networks in PyTorch

In this project, I am using a dataset provided by pybaseball. The dataset has over 700,000+ rows, containing every single pitch thrown in the 2023 MLB season as captured by Statcast. I aim to classify 17 pitch types by training a neural network in PyTorch that has 14 input features, 64 neurons in the first hidden layer, 32 neurons in the second hidden layer, and 17 classes in the output. Below, I provide the 14 features and their descriptions. The positive y-dimension points from the pitcher to the catcher, the positive x-dimension points to the catcher's right, and the positive y-dimension points upwards.

  * **release_spin**: spin rate of pitch tracked by Statcast (rpm)
  * **release_speed**: out-of-hand speed (MPH)
  * **release_extension**: release extension of pitch in feet as tracked by Statcast 
  * **release_pos_x**: horizontal release position of the ball measured in feet from the catcher's perspective
  * **release_pos_**: vertical release position of the ball measured in feet from the catcher's perspective
  * **spin_axis**: spin axis in the 2D X-Z plane in degrees from 0 to 360, such that 180 represents a pure backspin fastball and 0 degrees represents a pure topspin curveball
  * **vx0**: velocity of the pitch, in feet per second, in x-dimension, determined at y=50 feet
  * **vy0**: velocity of the pitch, in feet per second, in y-dimension, determined at y=50 feet
  * **vy0**: velocity of the pitch, in feet per second, in z-dimension, determined at y=50 feet
  * **pfx_x**: horizontal movement in feet from the catcher's perspective
  * **pfx_z**: vertical movement in feet from the catcher's perpsective
  * **p_throws**: pitcher's throwing hand ('R' or 'L')
  * **balls**: # of balls before the pitch is thrown
  * **strikes**: # of strikes before the pitch is thrown

In mlb_game_data.ipynb, I perform data processing, define the NeuralNetwork class, train several models using different optimizers, test the models using various metrics (accuracy, precision, recall), and finally attempt to manage dataset imbalance using the SMOTE and TomekLink techniques.
