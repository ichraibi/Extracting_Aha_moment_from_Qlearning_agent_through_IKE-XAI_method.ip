import os
import simplejson
import json
import numpy as np
from queue import Queue
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import networkx as nx
from itertools import islice
from graphviz import render
from sklearn.metrics import silhouette_score
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples

from networkx.drawing.nx_pydot import write_dot
import matplotlib.cm as cm

def get_dict_from_json(jsonfile):
    # data extracted from json are UNICODE
    dict = {}

    if os.path.exists(jsonfile) is True and os.path.isfile(jsonfile) is True:
        with open(jsonfile, 'r') as fp:
            dict = simplejson.load(fp)
    else:
        print("ERROR : The following file doesn't exist :" + jsonfile)
        print("Exiting the program...")
        exit()

    return dict

def get_part_of_dict(_data_dict, _nb_elements):
    if _nb_elements != 0:
        list_keys_dict = sorted(_data_dict.keys(), key=int)[:_nb_elements]
        new_dict = {}
        for a_key in list_keys_dict:
            new_dict[a_key] = _data_dict[a_key]
        return new_dict
    else:
        return _data_dict


def get_lists_of_labels_and_patterns_from_dict(_data_dict):
    list_labels = []
    list_patterns = []

    for a_key in sorted(_data_dict.keys(), key=int):
        # print("a_key:", a_key)
        list_labels.append(_data_dict[a_key]["input"])
        list_patterns.append(_data_dict[a_key]["hiddPattern"])

    return list_labels, list_patterns


def save_dict_in_json(dict, jsonfile):
    with open(jsonfile, 'w') as fp:
        json.dump(dict, fp, sort_keys=True, indent=4)

    return jsonfile


def initialize_dict_of_extracted_patterns(List_sequences, debug=False):
    dictionary = {}
    id_seq = 0

    print("Number of sequences to analyze : ", len(List_sequences))

    for a_seq in List_sequences:
        if isinstance(a_seq, list):
            one_seq = a_seq
            nb_samples = len(one_seq)-1
        else :
            one_seq = a_seq.split(";")
            assert((len(one_seq)-1)==a_seq.count(";"))
            nb_samples = a_seq.count(";")


        if debug:
            print("\n ---------------------------------------- ")
            print("id_seq : ", id_seq)
            print("one_seq : ", one_seq)

        if debug:
            print("nb_samples: ", nb_samples)

        for i in range(nb_samples):
            if debug:
                print("\ni: ", i)
                print("input          : ", one_seq[i])
                print("expected output: ", one_seq[i + 1])
                print("sequencePast:", one_seq[:i])

            dictionary[len(dictionary)] = {'input': one_seq[i],
                                           'desiredOutput': one_seq[i + 1],
                                           'idxSeq': int(id_seq),
                                           'oriSeq': one_seq,
                                           'erreur': 0,
                                           'hiddPattern': [],
                                           'idxSeq': id_seq,
                                           'net_output': [],
                                           'obtainedOutput': "",
                                           'sequencePast': one_seq[:i + 1]
                                           }

        id_seq += 1


    print("Size of the initialized dictionary : ", len(dictionary.keys()))
    if debug:
        print("Display of the first 100 elements of the dictionary  : ")
        for key in sorted(dictionary.keys())[:100]:
            print("\n key : ", key, " - val : ", dictionary[key])

    return dictionary

# Get and Save data for rules extraction (hidden states, predictions, sequences, ...) - TEST
def update_dict_of_extracted_patterns(dictionary, _predictions, _state_h, file, letter_dict, debug=False):
    
    if debug: print(len(dictionary), " - ",  len(_predictions), " - ", len(_state_h))
    assert len(dictionary) == len(_predictions) == len(_state_h)

    for i in range(len(dictionary.keys())):
        obtained_output_vector = _predictions[i][0]
        networkOutput = (obtained_output_vector == obtained_output_vector.max()).astype(float)
        obtained_output = get_letter_by_binary_value(networkOutput, letter_dict)

        if debug:
            print("obtained_output_vector : ", obtained_output_vector)
            print("networkOutput : ", networkOutput)
            print("obtained_output : ", obtained_output)
            print("dictionnary[i]['desired_output]' : ", dictionary[i]['desiredOutput'])
            #print("statePattern : ", _state_c.tolist()[i])
            print("hiddPattern : ", _state_h.tolist()[i])

            print("_predictions.tolist()[i][0] : ", _predictions.tolist()[i][0])
            print("type(_predictions.tolist()[i][0]) : ", type(_predictions.tolist()[i][0]))

            print("_state_h.tolist()[i] : ", _state_h.tolist()[i])
            print("type(_state_h.tolist()[i]) : ", type(_state_h.tolist()[i]))

            #print("_state_c.tolist()[i] : ", _state_c.tolist()[i])
            #print("type(_state_c.tolist()[i]) : ", type(_state_c.tolist()[i]))

        #dictionary[i]['aStatePattern'] = _state_c.tolist()[i]

        dictionary[i].update({'hiddPattern': _state_h.tolist()[i]}),

        dictionary[i].update({'net_output': _predictions.tolist()[i][0]}),

        dictionary[i]['obtainedOutput'] = obtained_output

        if debug :
            print("obtained_output : ", obtained_output, " -  desiredOutput : ", dictionary[i]['desiredOutput'])
            if obtained_output == dictionary[i]['desiredOutput']:
                print(" ---> OK ")
                dictionary[i]['erreur'] = 0
            else:
                print(" ---> KO ")
                dictionary[i]['erreur'] = 1

    print("Size of the initialed dictionnary : ", len(dictionary.keys()))
    if debug :
        print("Display of the 100 first elements: ")
        for key in sorted(dictionary.keys())[:100]:
            print("key : ", key, " - value : ", dictionary[key])

    save_dict_in_json(dictionary, file)
    print("\nSaving data for rules extraction (CEC, lstm_outputs and rnn_outputs) in file: ",
          file)

    return dictionary

def get_letter_by_binary_value(_a_binary_value, _letter_dict):
    theLetter = ""
    for KeyLetter, BinaryLetter in _letter_dict.items():
        if np.array_equal(BinaryLetter, _a_binary_value):
            theLetter = KeyLetter
            break
    return theLetter


def transform_sequences_into_data_for_rnn(symbol_dict, sequences, to_dictionary, debug =False):
    total_nb_samples = 0
    inputs = []
    expected_outputs = []
    dictionary = {}
    id_seq = len(dictionary)
    index = 0

    print("Number of sequences: ", len(sequences))

    if debug :
        for k, v in symbol_dict.items():
            print ("k : ", k, " - v :", v)

    for a_seq in sequences:
        if isinstance(a_seq, list):
            new_seq = a_seq
            nb_samples = len(new_seq)-1
        else :
            new_seq = a_seq.split(";")
            assert((len(new_seq)-1)==a_seq.count(";"))
            nb_samples = a_seq.count(";")

        total_nb_samples += nb_samples
        sequence_length = nb_samples

        if debug :
            print (" ")
            print("\n a_seq: ", a_seq)
            print("\n new_sequence: ", new_seq)
            print("\n id_sequence: ", id_seq)
            print("nb_samples: ", nb_samples)
            print("sequence_length: ", sequence_length)
            print("\n-------------------------------------------")

        for i in range(nb_samples):
            if debug:
                print("\ni: ", i)
                print("input          : ", new_seq[i], " ", symbol_dict[new_seq[i]])
                print("expected output: ", new_seq[i + 1], " ", symbol_dict[new_seq[i + 1]])

            inputs.append(symbol_dict[new_seq[i]])
            expected_outputs.append(symbol_dict[new_seq[i + 1]])



        if to_dictionary:
            # put data in dictionary for rules extraction
            dictionary[id_seq] = {'input': inputs[index:index + sequence_length],
                                  'expected_output': expected_outputs[index:index + sequence_length],
                                  'sequence': a_seq,
                                  'sequence_id': id_seq}
        id_seq += 1
        index += sequence_length

    if to_dictionary:
        return [inputs, expected_outputs, dictionary]
    else:
        return [inputs, expected_outputs]

def save_hidden_layer_states_and_predictions_in_json_file(jsonfile, predictions, state_h, state_c):
    dict = {}
    dict = {'obtained_outputs': predictions.tolist(), 'states_h': state_h.tolist(), 'states_c': state_c.tolist()}

    save_dict_in_json(dict, jsonfile)
    print("\nSaving data for rules extraction (CEC, lstm_outputs and rnn_outputs) in file: ", jsonfile)


# tools --------------------------------------------------------
def get_list_clusters_with_kmeans_and_silhouette_score_analysis(_clustering_range_min, _clustering_range_max, _list_patterns, debug):

    best_sil_score = 0
    n_clusters_for_best_sil_score = _clustering_range_min
    list_clusters_for_best_sil_score = []
    dict_silhouette_score_per_clusters = {}
    dict_res_kmeans_silhouette_analysis_details = {}
    for n_clusters in range(_clustering_range_min, _clustering_range_max):
        # K-means
        kmeans = KMeans(n_clusters=n_clusters)
        list_clusters = kmeans.fit_predict(_list_patterns)
        list_labels_clusters = kmeans.labels_

        # Silhouette-score to determine best value for n_clusters
        sil_score = silhouette_score(_list_patterns, list_clusters)

        if sil_score > best_sil_score:
            best_sil_score = sil_score
            n_clusters_for_best_sil_score = n_clusters
            list_clusters_for_best_sil_score = list_clusters

        dict_silhouette_score_per_clusters[n_clusters]={sil_score}
        if debug:
            print ( "score : ", sil_score, " - type(score) : ", type(sil_score))

            # the labels gives the index of the closest cluster each sample in X belongs to.
            # a sample belong to the cluster at the same position in the folowing list
            print ("kmeans.labels_ len: ", len(list_labels_clusters))
            print("kmeans.labels_ : ", list_labels_clusters)

        if debug:
            print ("\n kmean silhouette score analysis : -----------------")

        dict_res_kmeans_silhouette_analysis, silhouette_avg, dict_res_kmeans_silhouette_analysis_details = kmeans_silhouette_analysis(
            dict_res_kmeans_silhouette_analysis, np.asarray(_list_patterns), list_labels_clusters, n_clusters, kmeans,
            dict_res_kmeans_silhouette_analysis_details)
        setClusters = set(list_labels_clusters)
        int_setClusters = map(int, setClusters)

        if debug:
            print("setClusters : ", setClusters)
            print ("kmeans.inertia_ : ", kmeans.inertia_)
            print ("silhouette_avg : ", silhouette_avg)

            # cluster_centers_ : gives the coordinates of cluster centers
            print ("kmeans.cluster_centers_ len : ", len(kmeans.cluster_centers_))
            print ("kmeans.cluster_centers_ : ", kmeans.cluster_centers_)

        # inertia_ : float; Sum of distances of samples to their closest cluster center.
        a_inertia = kmeans.inertia_

        dict_res_kmeans_silhouette_analysis_details[n_clusters].update({'listClusters': list(int_setClusters), 'inertia': str(a_inertia), 'score': str(sil_score),'centers': kmeans.cluster_centers_.tolist()})

    print("\nThe best K value for Kmeans in range: [", _clustering_range_min, ";", _clustering_range_max, "] is k = ",
          n_clusters_for_best_sil_score, "with a silhouette_score value =", best_sil_score)


    save_dict_in_json(dict_res_kmeans_silhouette_analysis, "dict_kmeans_silhouette_values_per_cluster.json")
    save_dict_in_json(dict_res_kmeans_silhouette_analysis_details, "dict_kmeans_silhouette_values_per_cluster_detailled.json")

    draw_curve(dict_res_kmeans_silhouette_analysis.keys(), dict_res_kmeans_silhouette_analysis.values(), 'r')

    print("\nList of silhouette score values and the details of the kmeans algorithm saved in : dict_kmeans_silhouette_values_per_cluster.sjon and dict_kmeans_silhouette_values_per_cluster_detailled.json")

    return list_clusters_for_best_sil_score, n_clusters_for_best_sil_score

#===============================================================================================





def kmeans_silhouette_analysis( dict_res_kmeans_silhouette_analysis, data, cluster_labels, n_clusters, clusterer,
                               dict_res_kmeans_silhouette_analysis_details, debug):
    # src : http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    silhouette_avg = silhouette_score(data, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    dict_res_kmeans_silhouette_analysis[n_clusters] = silhouette_avg
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    dict_res_kmeans_silhouette_analysis_details[n_clusters].update(
        {'silhouette_avg': str(silhouette_avg), 'sample_silhouette_values': sample_silhouette_values.tolist(),
         'cluster_labels': cluster_labels.tolist()})

    dict_nb_data_per_cluster = {}
    y_lower = 10
    for i in range(n_clusters):

        dict_nb_data_per_cluster[i] = cluster_labels.tolist().count(i)

        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    dict_res_kmeans_silhouette_analysis_details[n_clusters].update(
        {'dict_nb_data_per_cluster': dict_nb_data_per_cluster})

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    imgPath = "KSilhouetteAnalysis_" + str(n_clusters)
    plt.savefig(imgPath)
    plt.close(fig)


    return dict_res_kmeans_silhouette_analysis, silhouette_avg, dict_res_kmeans_silhouette_analysis_details







def draw_curve(x, y, color, imgPath):

    plt.clf()
    plt.plot(x, y, color)
    plt.xlabel("Number of clusters")
    plt.ylabel("Average silhouette analysis")
    plt.title("The silhouette plot for the various clusters.")

    print ("img path : ", imgPath)
    plt.savefig(imgPath)


#===============================================================================================




def get_list_clusters_with_kmeans(_n_clusters, _list_patterns):
    kmeans = KMeans(n_clusters=_n_clusters)
    return kmeans.fit_predict(_list_patterns)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))



class finite_state_machine:
    def __init__(self, **attr):
        self.graph = nx.MultiDiGraph(**attr)
        self.transitions = self.graph.edges
        self.states = self.graph.nodes
        self.initial_state = None
        self.final_states = set()
        self.transition_functions = {}
        self.adj = self.graph.adj
        self.sink = None
        self.compact_view = True

    def add_state(self, state, initial=False, final=False, sink=False, **attr):
        shape = 'circle'
        penwidth = 1
        if initial:
            self.initial_state = state
            penwidth = 2
        if final:
            self.final_states.add(state)
            shape = 'doublecircle'
        self.transition_functions[state] = {}
        self.graph.add_node(state, initial=initial, final=final, sink=sink,
                            shape=shape, fixedsize=True, penwidth=penwidth, **attr)

    def add_transition(self, input_state, output_state, symbol, **attr):
        self.graph.add_edge(input_state, output_state, label=symbol, key=symbol, **attr)
        self.transition_functions[input_state].setdefault(symbol, []).append(output_state)

    def update_transition(self,input_state, output_state, symbol,_weight, **attr):
        self.graph.remove_edge(input_state, output_state)
        self.graph.add_edge(input_state, output_state, label=symbol, key=symbol,weight=_weight, **attr)

    def set_final_states(self, states, final=True):
        for state in states:
            self.graph.nodes[state]['final'] = final
            if final:
                self.final_states.add(state)
                self.graph.nodes[state]['shape'] = 'doublecircle'
            else:
                self.final_states.discard(state)
                self.graph.nodes[state]['shape'] = 'circle'

    def plot_simple_graph_with_heat_map(self, file_name, debug=False):
        print("\n\n Generation of fsm using heat map to display the transitions according their weights...")
        fsm= self.graph
        # creating a color list for each edge based on weight
        fsm_edges = fsm.edges(data=True)

        if debug: print("fsm_edges : ", fsm.edges(data=True))

        fsm_edges_weights = []
        fsm_edges_labels = {}
        i = 0

        for an_edge in fsm_edges:

            source, dest, attr_an_edge = an_edge[0], an_edge[1], an_edge[2]
            fsm_edges_weights.append(float(attr_an_edge['weight']))
            fsm_edges_labels[(an_edge[0], an_edge[1])] = {attr_an_edge['weight']}
            if debug:
                print("\n ---------------- ")
                print(source, " - ", dest, " - ", attr_an_edge)
                print(type(an_edge))
                print("an_edge : ", an_edge)
                print("type(attr_an_edge['weight']) : ", type(attr_an_edge['weight']))
                print("weight : ", attr_an_edge['weight'], " --end")
                # print("label : ", attr_an_edge['label'])
            i += 1

        # scale weights in range 0-1 before assigning color
        maxWeight = float(max(fsm_edges_weights))
        print("maxWeight : ", maxWeight)
        print("\n==================")
        fsm_colors = [plt.cm.winter_r(weight / maxWeight) for weight in fsm_edges_weights]

        # suppress plotting for the following dummy heatmap
        plt.ioff()

        # multiply all tuples in color list by scale factor
        colors_unscaled = [tuple(map(lambda x: maxWeight * x, y)) for y in fsm_colors]
        # generate a 'dummy' heatmap using the edgeColors as substrate for colormap
        heatmap = plt.pcolor(colors_unscaled, cmap=plt.cm.winter_r)

        # re-enable plotting
        # plt.ion()
        plt.clf()
        fig, axes = plt.subplots()
        ax = plt.gca()
        print(fsm)
        print(type(fsm))
        pos = nx.spring_layout(fsm.nodes())

        nodes = nx.draw_networkx_nodes(fsm, pos)
        # Set edge color to red
        nodes.set_edgecolor('black')
        nodes.set_linewidth(2)

        nx.draw_networkx(fsm, pos=pos, width=2, node_color='white', edge_color=fsm_colors, ax=axes, with_labels=True,
                         connectionstyle='arc3, rad = 0.1')

        # add colorbar
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel('edge weight', labelpad=15, rotation=270)
        # plt.show()

        plt.savefig(file_name)

    def plot_graph_with_heat_map(self, file_name, plot_label=True, debug = False):
        print ("\n\n Generation of fsm using heat map to display the transitions according their weights...")

        # creating a color list for each edge based on weight
        fsm_edges = self.graph.edges(data=True)

        if debug : print ("fsm_edges : ", self.graph.edges(data=True))

        fsm_edges_weights=[]
        fsm_edges_labels= {}
        for an_edge in fsm_edges :

            source, dest,attr_an_edge = an_edge[0],an_edge[1],an_edge[2]
            if debug : print(source, " - ", dest, " - ", attr_an_edge)
            fsm_edges_weights.append(attr_an_edge['weight'])
            fsm_edges_labels[(an_edge[0],an_edge[1])]={attr_an_edge['label']}
            if debug :
                print("\n ---------------- ")
                print(type(an_edge))
                print("weight : ",  attr_an_edge['weight'])
                print("label : ",  attr_an_edge['label'])


        # scale weights in range 0-1 before assigning color
        maxWeight = float(max(fsm_edges_weights))
        #fsm_colors = [plt.cm.Blues(weight / maxWeight) for weight in fsm_edges_weights] #another color
        fsm_colors = [plt.cm.winter_r(weight / maxWeight) for weight in fsm_edges_weights]

        # suppress plotting for the following dummy heatmap
        plt.ioff()

        # multiply all tuples in color list by scale factor
        colors_unscaled = [tuple(map(lambda x: maxWeight * x, y)) for y in fsm_colors]
        # generate a 'dummy' heatmap using the edgeColors as substrate for colormap
        heatmap = plt.pcolor(colors_unscaled, cmap=plt.cm.winter_r)

        # re-enable plotting
        if plot_label:
            plt.ion()
            fig, axes = plt.subplots()
            pos = nx.spring_layout(self.graph.nodes())
            nx.draw_networkx(self.graph, pos=pos, width=2, node_color='white', edge_color=fsm_colors, ax=axes,
                             with_labels=True,
                             connectionstyle='arc3, rad = 0.1')
            nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=fsm_edges_labels, label_pos=0.5, font_size=5,
                                         alpha=0.5)

        else:
            # re-enable plotting

            plt.clf()
            fig, axes = plt.subplots()
            plt.grid(False)
            ax = plt.gca()
            ax.set_facecolor("white")
            plt.rcParams["axes.edgecolor"] = "black"
            plt.rcParams["axes.linewidth"] = 1

            pos = nx.spring_layout(self.graph.nodes())

            nodes = nx.draw_networkx_nodes(self.graph, pos)
            # Set edge color to red
            nodes.set_edgecolor('black')
            nodes.set_linewidth(2)
            nx.draw_networkx(self.graph, pos=pos, width=2, node_color='white', edge_color=fsm_colors, ax=axes, with_labels=True,
                         connectionstyle='arc3, rad = 0.1')


            # add colorbar
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel('edge weight', labelpad=15, rotation=270)
        plt.savefig(file_name)

    def is_deterministic(self):
        for input_state, transition in self.transition_functions.items():
            for symbol, output_state in transition.items():
                if len(output_state) > 1:
                    return False
        return True

    def to_dfa(self):
        """Return a deterministic finite acceptor"""
        dfa = finite_state_machine()
        states_queue = Queue()

        dfa.add_state((self.initial_state,), initial=True)
        states_queue.put((self.initial_state,))

        while not states_queue.empty():

            dfa_state = states_queue.get()
            output_states = {}
            for state in dfa_state:
                for symbol in self.transition_functions[state].keys():
                    for output in self.transition_functions[state][symbol]:

                        output_states.setdefault(symbol, set()).add(output)

            for symbol, output_state in output_states.items():

                output_state = tuple(sorted(output_state, key=str))

                if output_state not in dfa.states().keys():
                    final = False
                    for state in output_state:
                        if self.is_final(state):
                            final = True
                            break
                    states_queue.put(output_state)
                    dfa.add_state(output_state, final=final)
                dfa.add_transition(dfa_state, output_state, symbol)
        return dfa


def create_fsm_object_from_dict(_fsm_dict, _fsm_name, _is_long_labels, debug=False):

    fsm = finite_state_machine(name=_fsm_name)

    if _is_long_labels is True:
        label = 'long_label'
        print("\n Creation of fsm from dictionnary with long labels")
    else :
        label = 'short_label'
        print("\n Creation of fsm from dictionnary with short labels")

    for a_node in _fsm_dict['nodes']:
        if a_node == "-1":
            fsm.add_state(a_node, initial=True)
        else:
            fsm.add_state(a_node)

    list_final_nodes = []

    for a_transition in _fsm_dict['transitions']:
        if debug :
            print("a_transition['id'][0] : ", a_transition['id'][0])
            print("a_transition['id'][1] : ", a_transition['id'][1])
            print("a_transition[label] : ", a_transition[label])
            print("a_transition['weight'] : ", a_transition['weight'])

        fsm.add_transition(a_transition['id'][0], a_transition['id'][1], a_transition[label])
        fsm.update_transition(a_transition['id'][0], a_transition['id'][1], a_transition[label], a_transition['weight'])

        if _is_long_labels is False :
            if "B" in a_transition[label] and a_transition['id'][0] != "-1":
                list_final_nodes.append(a_transition['id'][0])

    fsm.set_final_states(list_final_nodes)

    return fsm


def save_fsm_graph_in_dot_format(_fsm_object, _dot_filename):
    # https://networkx.github.io/documentation/stable/reference/drawing.html
    write_dot(_fsm_object.graph, _dot_filename)
    print("FSM graph saved (dot format) in file :", _dot_filename)


def convert_dot_graph_to_png(_dot_graph_filename):
    # https://graphviz.readthedocs.io/en/stable/manual.html#existing-files
    render('dot', 'png', _dot_graph_filename)
    print("Dot graph", _dot_graph_filename, "converted in PNG format : ", _dot_graph_filename+".png")


def create_png_for_graph_from_dot(_fsm_object, _filename):
    write_dot(_fsm_object.graph, _filename)
    render('dot', 'png', _filename)
    print("Creation of files: ", _filename+".dot", " and ", _filename+".png")



def knowledge_extraction_process(original_data_dict, clustering_range_min, clustering_range_max, #pattern, label,
                                 nb_patterns,  dir_result="",debug=False):
    # ----------------Retreives patterns and labels from the data file to extract knowledge in the shape of automata-------------#

    data_dict = get_part_of_dict(original_data_dict, nb_patterns)

    list_labels, list_patterns = get_lists_of_labels_and_patterns_from_dict(data_dict)
    if debug :
        print("len data_dict:", len(data_dict))
        print("len list_patterns", len(list_patterns))
        # print("list_patterns", list_patterns)
        print("len list_labels", len(list_labels))
        # print("list_labels", list_labels)


    dict_silhouette_scores = {}

    # ----------Clustering---------------#
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    for kmeans_val in range(clustering_range_min, clustering_range_max):

        list_clusters = get_list_clusters_with_kmeans(kmeans_val, list_patterns)
        sil_score = silhouette_score(list_patterns, list_clusters)
        dict_silhouette_scores[kmeans_val] = sil_score

        if debug:
            print("list_clusters: ", list_clusters)
            print("len list_clusters: ", len(list_clusters))
            print("list_clusters: ", list_clusters)
            print("len list_clusters: ", len(list_clusters))

        # --------------Automate-------------#
        # json
        fsm_dict = {}
        current_node = str(-1)
        fsm_dict['nodes'] = [current_node]
        fsm_dict['transitions'] = []

        for i in range(len(list_patterns)):
           
            linked_cluster = str(list_clusters[i])

            if linked_cluster not in fsm_dict['nodes']:
                fsm_dict['nodes'].append(linked_cluster)

            current_transition = {}
            current_transition['id'] = [current_node, linked_cluster]
            transition_is_in_dict = False

            index = 0
            for a_transition_from_dict in fsm_dict['transitions']:
                if current_transition['id'] == a_transition_from_dict['id']:
                    transition_is_in_dict = True
                    break;
                index += 1

            transition_label = list_labels[i] + "_" + str(i)

            if transition_is_in_dict is False:
                new_transition = current_transition
                new_transition['weight'] = 1
                new_transition['long_label'] = transition_label
                new_transition['short_label'] = list_labels[i]
                fsm_dict['transitions'].append(new_transition)
            else:
                fsm_dict['transitions'][index]['weight'] += 1
                fsm_dict['transitions'][index]['long_label'] += ";" + transition_label
                if list_labels[i] not in fsm_dict['transitions'][index]['short_label']:
                    fsm_dict['transitions'][index]['short_label'] += ";" + list_labels[i]

            current_node = linked_cluster

        if debug: print("\nFSM dict (two first element in dictionnary) : ", take(2, fsm_dict.items()))

        # filenames
        directory = dir_result+"Graphs/"

        if not os.path.exists(directory):
            os.mkdir(directory)

        fsm_long_labels_json = directory + "fsm_kmeans_" + str(kmeans_val) + ".json"
        fsm_long_labels_dot = directory + "fsm_long_labels_kmeans_" + str(kmeans_val)
        fsm_short_labels_dot = directory + "fsm_short_labels_kmeans_" + str(kmeans_val)
        dfa_file = directory + "dfa_kmeans_" + str(kmeans_val)
        complete_dfa_file = directory + "complete_dfa_kmeans_" + str(kmeans_val)
        min_dfa_file = directory + "min_dfa_kmeans_" + str(kmeans_val)
        accepted_seq_dict_file = directory + "rg_accepted_seq_dict.json"
        random_accepted_seq_dict_file = directory + "random_accepted_seq_dict.json"
        fsm_long_labels_png = directory + "fsm_long_labels_kmeans_png_" + str(kmeans_val) + ".png"
        fsm_short_labels_png = directory + "fsm_short_labels_kmeans_png_" + str(kmeans_val) + ".png"

        # LONG LABELS
        # save fsm dict in json file
        save_dict_in_json(fsm_dict, fsm_long_labels_json)
        print("\nFSM dict saved in json file:", fsm_long_labels_json)

        fsm_long_labels = create_fsm_object_from_dict(fsm_dict, "FSM", True, debug)
        # save fsm in dot and png files
        create_png_for_graph_from_dot(fsm_long_labels, fsm_long_labels_dot)
        fsm_long_labels.plot_graph_with_heat_map(fsm_long_labels_png, debug)

        # # SHORT LABELS
        fsm_short_labels = create_fsm_object_from_dict(fsm_dict, "FSM", False, debug)
        # save fsm in dot and png files
        create_png_for_graph_from_dot(fsm_short_labels, fsm_short_labels_dot)
        fsm_short_labels.plot_graph_with_heat_map(fsm_short_labels_png, debug)
        fsm_short_labels.plot_graph_with_heat_map(fsm_short_labels_png.replace(".png","_no_label.png"), False, debug)


    save_dict_in_json(dict_silhouette_scores, dir_result+"dict_silhouette_scores.json")

