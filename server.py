'''
You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

NAME :Junyi Tao
ID :112820617
DATE :
HOMEWORK :

'''

from flask import Flask, request
from flask import render_template
import time
import json
from scipy.interpolate import interp1d
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])

#compute the total distance for each word (based on center)
def com_distance(points_X, points_Y):
    distance,disl,cul_disl=0,[],[]
    x_ch,y_ch=[],[]
    for i in range(len(points_X)-1):
        cur_x,cur_y=points_X[i], points_Y[i]
        next_x,next_y=points_X[i+1], points_Y[i+1]
        tmp_distance=math.sqrt((next_x-cur_x)**2+(next_y-cur_y)**2)
        distance+=tmp_distance
        cul_disl.append(distance)
        disl.append(tmp_distance)
        x_ch.append(next_x-cur_x)
        y_ch.append(next_y-cur_y)
    return distance,disl,cul_disl,x_ch,y_ch

def issame(p):
   meanp = sum(p) / len(p)
   for pp in p:
      if abs(pp - meanp) > 1e-6:
        return False
   return True
    
def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    # TODO: Start sampling
    sample_points_X, sample_points_Y=[],[]
    evenly_spaced_100_numbers = np.linspace(0, 1, 100)
    cur_point=0
    if (len(points_X)==1 or (issame(points_X) and issame(points_Y))):
                   return [points_X[0]] * 100, [points_Y[0]] * 100
    else:
       total_len,dis_list,cul_disl,x_ch,y_ch=com_distance(points_X, points_Y)
       ratio=np.array(cul_disl)/total_len
       for i in range(len(dis_list)):
           while(evenly_spaced_100_numbers[cur_point]<ratio[i]):
                 if (i>0):
                        tmp_distance=total_len*(evenly_spaced_100_numbers[cur_point]-ratio[i-1])
                 else:
                        tmp_distance=total_len*evenly_spaced_100_numbers[cur_point]
                    
                 temp_x=tmp_distance*x_ch[i]/dis_list[i]+points_X[i]
                 temp_y=tmp_distance*y_ch[i]/dis_list[i]+points_Y[i]
                 sample_points_X.append(temp_x)
                 sample_points_Y.append(temp_y)
                 cur_point+=1
    sample_points_X.append(points_X[-1])
    sample_points_Y.append(points_Y[-1])
    return sample_points_X, sample_points_Y
    
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    #I don't compute the probabilities of valid words
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 15#Enter Value Here
    
    gest_startx,gest_starty=gesture_points_X[0], gesture_points_Y[0]
    gest_endx,gest_endy=gesture_points_X[-1], gesture_points_Y[-1]
    for i in range(len(template_sample_points_X)):
        cur_X, cur_Y=template_sample_points_X[i], template_sample_points_Y[i]
        cur_startx,cur_starty=cur_X[0], cur_Y[0]
        cur_endx,cur_endy=cur_X[-1], cur_Y[-1]
        start_dis=math.sqrt((cur_startx-gest_startx)**2+(cur_starty-gest_starty)**2)
        end_dis=math.sqrt((cur_endx-gest_endx)**2+(cur_endy-gest_endy)**2)
        if(start_dis<=threshold and end_dis<=threshold):
           valid_words.append(words[i])
           valid_template_sample_points_X.append(template_sample_points_X[i])
           valid_template_sample_points_Y.append(template_sample_points_Y[i])
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y

def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum
    r = L/max(H, W)
    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X))/2
    centroid_y = (max(gesture_Y) - min(gesture_Y))/2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)

    return scaled_X, scaled_Y
    

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores=[]
    # TODO: Set your own L
    L = 1.0 #Enter Value Here
    # TODO: Calculate shape scores
    #s_gesx,s_gesy=get_scaled_points(gesture_sample_points_X, gesture_sample_points_Y, L)
    for i in range(len(valid_template_sample_points_X)):
        X,Y=valid_template_sample_points_X[i], valid_template_sample_points_Y[i]
        s_X,s_Y=get_scaled_points( X,Y, L)
        total_distance=0
        for j in range(len(X)):
           total_distance+=math.sqrt((s_X[j]-gesture_sample_points_X[j])**2+(s_Y[j]-gesture_sample_points_Y[j])**2)
           #total_distance+=math.sqrt((s_X[j]-s_gesx[j])**2+(s_Y[j]-s_gesy[j])**2)
           #total_distance+=math.sqrt((X[j]-gesture_sample_points_X[j])**2+(Y[j]-gesture_sample_points_Y[j])**2)
        shape_scores.append(total_distance / len(X))
    return shape_scores
                
def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # TODO: Calculate location scores
    # compute alphas
    alphas = np.zeros((100))
    for i in range(50):
        x = i/2550
        alphas[50 - i - 1], alphas[50 + i] = x, x
        
    location_scores = np.zeros((len(valid_template_sample_points_X)))
    # Create a list of gesture points [[xi, yi]]
    gesture_points = [[gesture_sample_points_X[j], gesture_sample_points_Y[j]] for j in range(100)]
    for i in range(len(valid_template_sample_points_X)):
        template_points = [[valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j]] for j in range(100)]
        distances = euclidean_distances(gesture_points, template_points)
        template_gesture_min = np.min(distances, axis=0)
        gesture_template_min = np.min(distances, axis=1)
        if np.any(gesture_template_min-radius>0) or np.any(template_gesture_min-radius>0 ):
            deltas = np.diagonal(distances)
            location_scores[i] = np.sum(np.multiply(alphas, deltas))
    return location_scores

#how to set
def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 2/3#Enter Value Here#
    # TODO: Set your own location weight
    location_coef = 1/3#Enter Value Here#
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 3#Enter Value Here
    # Find indices having the minimum score
    best_idx = np.argsort(integration_scores)[0]
    best_word = valid_words[best_idx]
    # Return the best words separated by space
    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    #gesture_points_X = [gesture_points_X]
    #gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)
    
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)
    #valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_sample_points_X, gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y)
    
    best_word = "Word not found"
    
    if len(valid_words) != 0:
       shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

       location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

       integration_scores = get_integration_scores(shape_scores, location_scores)

       best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()
    
    print('{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}')

    return '{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}'


if __name__ == "__main__":
    app.run()
