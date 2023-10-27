###
# Multiple Object Tracking Simulation
#
# This simulation has various aspects to simulate mulitple target tracking using the Kalman filter
# such as generating detections, clutter, and ground truths. It is based on the following link:
# https://stonesoup.readthedocs.io/en/latest/auto_tutorials/06_DataAssociation-MultiTargetTutorial.html
#
# Peri Hassanzadeh
# Last Modified: 10/27/23
###

#Import Libraries
import numpy as np
from datetime import datetime, timedelta
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.plotter import AnimatedPlotterly
from stonesoup.types.detection import Clutter, TrueDetection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from ordered_set import OrderedSet
from scipy.stats import uniform

#Current Time
start_time=datetime.now().replace(microsecond=0)

def genGT(genDetections, genClutter, KalmanAssociation):
	np.random.seed(1991)
	truths = OrderedSet()
	transition_model= CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005), ConstantVelocity(0.005)])

	#Starts Time at Current Time
	timesteps=[start_time]

	#First Target: Starts at (0,0) 
	truth=GroundTruthPath([GroundTruthState([0,1,0,2], timestamp=timesteps[0])])
	for k in range(1,21):
		timesteps.append(start_time+timedelta(seconds=k))
		truth.append(GroundTruthState(transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)), timestamp=timesteps[k]))

	truths.add(truth)

	#Second Target: Starts at (0, 20)
	truth = GroundTruthPath([GroundTruthState([0,1,20,-1], timestamp=timesteps[0])])
	for k in range(1,21):
		truth.append(GroundTruthState(transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)), timestamp=timesteps[k]))

	_ = truths.add(truth)

	plotter = AnimatedPlotterly(timesteps, tail_length=0.1)
	plotter.plot_ground_truths(truths, [0,2])
	plotter.fig


	########### Generate Detections with Clutter ##########################
	if genDetections==True:
		measurement_model = LinearGaussian(ndim_state=4, mapping=(0,2), noise_covar=np.array([[0.75, 0],[0,0.75]]))
		all_measurements = []

		#Iterate through 20 timestamps on plot
		for k in range(20):
			measurement_set=set()

			for truth in truths:
				if np.random.rand() <=0.9:
					measurement = measurement_model.function(truth[k], noise=True)
					measurement_set.add(TrueDetection(state_vector=measurement, groundtruth_path=truth, timestamp=truth[k].timestamp, measurement_model=measurement_model))

				truth_x = truth[k].state_vector[0]
				truth_y = truth[k].state_vector[2]

				#Generating Clutter
				if genClutter==True:
					for _ in range(np.random.randint(10)):
						x = uniform.rvs(truth_x-10, 20)
						y = uniform.rvs(truth_y-10, 20)
						measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp, measurement_model=measurement_model))

			all_measurements.append(measurement_set)

		plotter.plot_measurements(all_measurements, [0,2])
	#plotter.fig.show()

	####### Kalman Predictor and Updater & Run Kalman Filter
	if KalmanAssociation==True:
		predictor = KalmanPredictor(transition_model)
		updater = KalmanUpdater(measurement_model)

		hypothesiser=DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

		dataassociator = GlobalNearestNeighbour(hypothesiser)

		prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
		prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

		tracks = {Track([prior1]), Track([prior2])}

		for n, measurements in enumerate(all_measurements):
			hypotheses = dataassociator.associate(tracks, measurements, start_time+timedelta(seconds=n))

			for track in tracks:
				hypothesis = hypotheses[track]
				if hypothesis.measurement:
					post=updater.update(hypothesis)
					track.append(post)
				else:
					track.append(hypothesis.prediction)
		
		plotter.plot_tracks(tracks,[0,2], uncertainty=True)

	plotter.fig.show()


def main():
	genGT(genDetections=True, genClutter=True, KalmanAssociation=True)


if __name__ == "__main__":
	main()