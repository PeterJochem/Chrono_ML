# Datasets Generated in Chrono
These are our datasets of Chrono data. A description of each data set is below. 

# dset1
csv file of Gamma, Beta, Depth, Position X, Position Z, GRF X, GRF Z <br />

# dset2
This dataset is the same as the first dataset but it includes the velocity information. The velocities are very small though because the RFT model does not depend on the velocity of the plate, the data was generated at low speeds. The neural network's predictions do depend on the velocity of the plate because the underlying physics depend on the velocity. <br />

csv file of Gamma, Beta, Depth, Position X, Position Z, Velocity X, Velocity Z, GRF X, GRF Z

# dset3
This includes variations in the speed at which the plate traverses through the granular material. The speeds are also in the range of what Dan's robot will be moving at (0.1 - 1.0) m/s <br />

csv file of Gamma, Beta, Depth, Position X, Position Y, Position Z, Velocity X, Velocity Y, Velocity Z, GRF X, GRF Y, GRF Z 

