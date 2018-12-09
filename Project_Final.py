import networkx as nx
import numpy as np
import random
from random import shuffle
import math
import operator
import matplotlib.pyplot as plt

def GraphGeneration(n,p,st):
    #n = number of individuals in the network, make sure that this number is divisible by 100
    #p = percent of individuals on prep
    #This should implement the configuration model given the degree distributions.

    G = nx.Graph()
    nodes_list = range(0,n)

    G.add_nodes_from(nodes_list)

    degree = [0,1,2,3,4,5,6,7,8,9,10]
    prob = [5,21,10,9,9,8,8,8,8,8,6]

    nodes_avail = nodes_list
    shuffle(nodes_avail)
    nodes_list = range(0, n)
    boxes = [] #This is a list of nodes that repeat according to their degree

    idx = 0

    for deg in degree:
        prob_deg = prob[idx]
        n_nodes_deg = n*prob_deg/100
        nodes_deg = nodes_avail[0:n_nodes_deg-1]
        if deg == 0: #This deletes nodes that do not have any contact with other people from the node
            for nod in nodes_deg:
                nodes_avail.remove(nod)
        else:
            for nod in nodes_deg:

                for j in range(0,deg):
                    boxes.append(nod)
                nodes_avail.remove(nod)


    shuffle(boxes)

    #This part generates edges for the nodes
    half_val = len(boxes)/2

    for i in range(0,half_val):
        G.add_edge(boxes[i],boxes[i+half_val])

    #This following sections defines the initial attributes of the network
    #Node_attributes = HIV Status and treatment status, Prep Status, Gonorrhea Status, Gonorrhea Age
    #Edge_attributes = Steadiness, probability of HIV, probabilty of gonorrhea, length of relationship

    #Initial parameters

    hiv_shuffle = range(0,n)
    shuffle(hiv_shuffle)

    hiv_infected_nt = 0.072*0.24
    n_hivinfected_nt = math.floor(n*hiv_infected_nt)


    hiv_infected_t = 0.072*0.76
    n_hivinfected_t = math.floor(n*hiv_infected_t)



    n_onprep = math.floor(n*p)


    n_susceptible = n - n_onprep - n_hivinfected_nt - n_hivinfected_t


    hiv_status = {}
    hiv_type = {}


    #Generate dictionary of hiv status, 1 = susceptible on prep, 2 = susceptible not on prep, 3 = infected on treatment, 4 = infected not on treatment
    for i in range(0,int(n_hivinfected_nt)):
        hiv_status.update({hiv_shuffle[i]:4})
        hiv_type.update({hiv_shuffle[i]:0})

    counted = n_hivinfected_nt + n_hivinfected_t

    for i in range(int(n_hivinfected_nt),int(counted)):
        hiv_status.update({hiv_shuffle[i]:3})
        hiv_type.update({hiv_shuffle[i]: 0})

    counted = counted + n_susceptible


    for i in range(int(counted-n_susceptible),int(counted)):
        hiv_status.update({hiv_shuffle[i]:2})
        hiv_type.update({hiv_shuffle[i]: 0})

    for i in range(int(counted),n):
        hiv_status.update({hiv_shuffle[i]:1})
        hiv_type.update({hiv_shuffle[i]: 0})

    nonprep = 0
    nsusceptible = 0
    for nod in hiv_status:
        if hiv_status[nod] == 1:
            nonprep = nonprep + 1
        if hiv_status[nod] == 2:
            nsusceptible = nsusceptible + 1

    nx.set_node_attributes(G,'hiv_status',hiv_status)
    nx.set_node_attributes(G,'hiv_type',hiv_type)

    gonorrhea_inf = 0.06
    n_goninfected = math.floor(n*gonorrhea_inf)

    gonorrhea_sus = 0.94
    n_gonsusceptible = math.floor(n*gonorrhea_inf)

    gon_shuffle = range(0,n)
    shuffle(gon_shuffle)

    #Generates a dictionary fo gonorrhea status, 0 is absence of disease, 1 is infection
    gon_status = {}
    gon_age = {}
    gon_type = {} #0 is the person is not infected, 1 is while being on prep, and 2 is while not on pre


    for i in range(0,int(n_goninfected)):
        gon_status.update({gon_shuffle[i]:1})
        gon_age.update({gon_shuffle[i]:1})
        if hiv_status[gon_shuffle[i]] > 1:
            gon_type.update({gon_shuffle[i]:2})
        elif hiv_status[gon_shuffle[i]] == 1:
            gon_type.update({gon_shuffle[i]: 1})

    for i in range(int(n_goninfected),n):
        gon_status.update({gon_shuffle[i]:0})
        gon_age.update({gon_shuffle[i]: 0})
        gon_type.update({gon_shuffle[i]: 0})

    nx.set_node_attributes(G, 'gon_type', gon_status)
    nx.set_node_attributes(G, 'gon_status', gon_status)
    nx.set_node_attributes(G, 'gon_age', gon_age)

    n_edges = G.number_of_edges()

    stead_prop = st
    n_steady = math.floor(stead_prop*n)

    edges_shuffle = G.edges()
    shuffle(edges_shuffle)
    #Casual is defined as 0 and steady as 1

    relationship_status = {}
    for i in range(0,int(n_steady)):
        relationship_status.update({edges_shuffle[i]:1})


    for i in range(int(n_steady),n_edges):
        relationship_status.update({edges_shuffle[i]:0})

    nx.set_edge_attributes(G,'relationship_type',relationship_status)

    G = HIV_probability(G)
    G = gon_probability(G)

    age = {}
    for edg in G.edges():
        age.update({edg:0})

    nx.set_edge_attributes(G, 'relationship_age', age)

    print 'Initial Graph Generated'
    return G,p


def HIV_probability(G):


    probSI = 0.137
    probPI = probSI*0.24
    hiv_status = nx.get_node_attributes(G,'hiv_status')

    edges = G.edges()

    hiv_prob = {}

    for edg in edges:
        ini = edg[0]
        fin = edg[1]

        if (hiv_status[ini] == 2 and hiv_status[fin] > 3) or (hiv_status[ini] > 3 and hiv_status[fin] == 2):
            hiv_prob.update({edg:probSI})
        elif (hiv_status[ini] == 1 and hiv_status[fin] == 4) or (hiv_status[ini] == 4 and hiv_status[fin] == 1):
            hiv_prob.update({edg:probPI})
        else:
            hiv_prob.update({edg:0})

    prob1 = 0
    prob2 = 0
    prob3 = 0
    for pro in hiv_prob:
        if hiv_prob[pro] == probPI:
            prob2 = prob2 + 1
        if hiv_prob[pro] == probSI:
            prob1 = prob1 + 1
        else:
            prob3 = prob3 + 1

    nx.set_edge_attributes(G, 'hiv_prob', hiv_prob)

    return G

def gon_probability(G):

    prob_prep = 0.13
    prob_suc = 0.10
    gon_status = nx.get_node_attributes(G, 'gon_status')
    hiv_status = nx.get_node_attributes(G, 'hiv_status')
    n_prep = 0
    n_suc = 0

    n_prep_vul = []
    n_suc_vul = []

    for nod in gon_status:
        if hiv_status[nod] == 1:
            n_prep = n_prep + 1
        if hiv_status[nod] == 2:
            n_suc = n_suc + 1

    for edg in G.edges():
        ini = edg[0]
        fin = edg[1]

        if hiv_status[ini] == 2 and gon_status[fin] == 1:
            n_suc_vul.append(ini)
        if gon_status[ini] == 1 and hiv_status[fin] == 2:
            n_suc_vul.append(fin)
        if hiv_status[ini] == 1 and gon_status[fin] == 1:
            n_prep_vul.append(ini)
        if gon_status[ini] == 1 and hiv_status[fin] == 1:
            n_prep_vul.append(fin)

    n_prep_vul = set(n_prep_vul)
    n_suc_vul = set(n_suc_vul)
    n_prepvul = len(n_prep_vul)
    n_sucvul = len(n_suc_vul)

    prep_infected = math.floor(n_prep * prob_prep) + 1
    suc_infected = math.floor(n_suc * prob_suc) + 1
    if n_sucvul == 0:
        n_sucvul = 1
    if n_prepvul == 0:
        n_prepvul = 1

    probSI = suc_infected / n_sucvul
    probPI = prep_infected / n_prepvul

    if probSI > 1:
        probSI = 1

    if probPI > 1:
        probPI = 1

    edges = G.edges()

    gon_prob = {}

    for edg in edges:
        ini = edg[0]
        fin = edg[1]

        if (hiv_status[ini] == 2 and gon_status[fin] == 1) or (gon_status[ini] == 1 and hiv_status[fin] == 2):
            gon_prob.update({edg: probSI})
        elif (hiv_status[ini] == 1 and gon_status[fin] == 1) or (gon_status[ini] == 1 and hiv_status[fin] == 1):
            gon_prob.update({edg: probPI})
        else:
            gon_prob.update({edg: 0})

    nx.set_edge_attributes(G, 'gon_prob', gon_prob)

    return G

def NewEdgesGeneration(G,st):

    #Degree distribution that has to be kept constant

    degreev = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    prob = [5, 21, 10, 10, 9, 9, 9, 9, 9, 9]
    nodes_deg = []

    node_list = G.nodes()
    idx = 0
    for deg in degreev:
        prob_deg = prob[idx]
        idx = idx+1
        n_nodes_deg =  n * prob_deg / 100
        nodes_deg.append(n_nodes_deg)

    dict_degree = {}
    degrees =  []
    for nod in node_list:
        deg = G.degree(nod)
        dict_degree.update({nod:deg})
        degrees.append(deg)

    new_dict_degree = dict_degree.copy()

    deg_dist_cur = []

    for deg in degreev:
        n_deg = degrees.count(deg)
        deg_dist_cur.append(n_deg)

    for i in range(1,len(degreev)):
        target_deg = degreev[-i]
        n_target = nodes_deg[-i]
        n_current = deg_dist_cur[-i]
        for k in range(1,target_deg+1):
            deg_iter = target_deg - k
            for nod in node_list:
                if new_dict_degree[nod] == deg_iter:
                    new_dict_degree[nod] = target_deg
                    n_current = n_current + 1

                    deg_dist_cur[deg_iter] = deg_dist_cur[deg_iter] - 1
                if n_current == n_target:
                    break
            if n_current == n_target:
                break

    dif_dict = {}


    for key in dict_degree:
        dif = new_dict_degree[key]-dict_degree[key]
        dif_dict.update({key:dif})

    creator_list = []

    for key in dif_dict:
        times = dif_dict[key]
        count = 0
        while count < times:
            creator_list.append(key)
            count = count + 1

    shuffle(creator_list)
    n_newedges = len(creator_list)/2

    #Create new deges

    rel_type = nx.get_edge_attributes(G,'relationship_type')
    rel_age = nx.get_edge_attributes(G,'relationship_age')

    for i in range(0,n_newedges):
        dice = np.random.rand(1)
        if dice < st:
            G.add_edge(creator_list[i],creator_list[i+n_newedges])
            rel_type.update({(creator_list[i],creator_list[i+n_newedges]):1})
            rel_age.update({(creator_list[i],creator_list[i+n_newedges]):0})
        else:
            G.add_edge(creator_list[i], creator_list[i + n_newedges])
            rel_type.update({(creator_list[i], creator_list[i + n_newedges]): 0})
            rel_age.update({(creator_list[i], creator_list[i + n_newedges]): 0})

    nx.set_edge_attributes(G,'relationship_type',rel_type)
    nx.set_edge_attributes(G,'relationship_age', rel_age)

    degrees1 =[]
    for nod in node_list:
        deg1 = G.degree(nod)
        degrees1.append(deg1)

    deg_dist_fin=[]
    for deg in degreev:
        n_deg = degrees1.count(deg)
        deg_dist_fin.append(n_deg)
    print deg_dist_fin
    #Compute HIV and Gonorrhea probabilities:

    G = gon_probability(G)
    G = HIV_probability(G)


    return G

def AdvancingRelations(G):
    '''This function computes if an edge is broken or if it fortifies '''
    age = nx.get_edge_attributes(G,'relationship_age')

    #Updating age
    for age_val in age:
        ini_val = age[age_val]
        age[age_val] = ini_val+1

    #Deleting or creating edge
    statush = nx.get_edge_attributes(G,'relationship_type')

    for edg in G.edges():
        age_rel = age[edg]
        statusi = statush[edg]
        prob = prob_breakup(statusi,age_rel)
        dice = np.random.rand(1)
        ini = edg[0]
        fin = edg[1]
        if dice <= prob:
            G.remove_edge(ini,fin)


    return G

def prob_breakup(index,age):

    prob_casual = [0.77,0.32,0.24,0.32,0.14,0.17]
    prob_steady = [0.46,0.27,0.19,0.24,0.17,0.21]

    if index == 0:
        if age <= 6:
            prob = prob_casual[age-1]
        else:
            prob = 0.1

    if index == 1:
        if age <= 6:
            prob = prob_steady[age-1]
        else:
            prob = 0.1

    return prob


def HIV_Spreading(G):
    ''' Updates attribute of HIV status according to the status of the network'''

    n = G.number_of_edges()
    dices =  np.random.rand(1,n)
    prob_hiv = nx.get_edge_attributes(G,'hiv_prob')
    type_relation = nx.get_edge_attributes(G,'relationship_type')

    hiv_status = nx.get_node_attributes(G,'hiv_status')
    hiv_type = nx.get_node_attributes(G,'hiv_type')
    counter = 0

    for edg in G.edges():
        ini = edg[0]
        fin = edg[1]
        prob = prob_hiv[edg]
        type = type_relation[edg]
        val = dices[0]
        val = val[counter]
        rela = 0
        if val <= prob:

            if type == 0:
                rela = 1
            elif type == 1:
                rela =2
            if hiv_status[ini] == 4:
                hiv_type[fin] = rela
            elif hiv_status[fin] == 4:
                hiv_type[ini] = rela

            treatment = np.random.rand(1)
            if treatment <= 0.24:
                hiv_status[ini] = 4
                hiv_status[fin] = 4
            elif treatment > 0.24:
                if hiv_status[ini] != 3:
                    hiv_status[ini] = 4
                if hiv_status[fin] != 3:
                    hiv_status[fin] = 4
        counter = counter + 1

    nx.set_node_attributes(G, 'hiv_status', hiv_status)
    nx.set_node_attributes(G, 'hiv_type', hiv_type)
    return G

def Gonorrhea_Spreading(G):
    ''' Updates attribute of Gonorrhea status according to the status of the network'''
    n = G.number_of_edges()
    dices = np.random.rand(1, n)
    prob_gon = nx.get_edge_attributes(G, 'gon_prob')
    gon_age = nx.get_node_attributes(G, 'gon_age')
    gon_status = nx.get_node_attributes(G, 'gon_status')
    gon_type = nx.get_node_attributes(G, 'gon_type')
    for nod in gon_status:
        if gon_status[nod] == 1:
            gon_age[nod] = gon_age[nod] + 1
    hiv_status = nx.get_node_attributes(G, 'hiv_status')
    counter = 0

    for edg in G.edges():
        ini = edg[0]
        fin = edg[1]
        prob = prob_gon[edg]
        val = dices[0]
        val = val[counter]
        if val <= prob:
            gon_status[ini] = 1
            gon_status[fin] = 1
            if hiv_status[ini] == 1 and  gon_type[ini] == 0:
                gon_type[ini] = 1
            elif hiv_status[ini] > 1 and gon_type[ini] == 0:
                gon_type[ini] = 2

            if hiv_status[fin] == 1 and  gon_type[fin] == 0:
                gon_type[fin] = 1
            elif hiv_status[fin] > 1 and gon_type[fin] == 0:
                gon_type[fin] = 2

            if gon_age[ini] == 0:
                gon_age[ini] = 1
            elif gon_age[fin] == 0:
                gon_age[fin] =1

        counter = counter + 1

    nx.set_node_attributes(G, 'gon_type', gon_type)
    nx.set_node_attributes(G,'gon_status',gon_status)
    nx.set_node_attributes(G, 'gon_age', gon_age)
    return G

def TreatHIV(G,pr):

    hiv_status = nx.get_node_attributes(G,'hiv_status')
    hiv_nt = 0
    for nod in hiv_status:
        if hiv_status[nod] == 4:
            hiv_nt = hiv_nt + 1
    nt_to_t = math.floor(hiv_nt*pr)

    count = 0
    for nod in hiv_status:
        if hiv_status[nod] == 4 and count<= nt_to_t:
            hiv_status[nod] = 3
            count = count + 1

    nx.set_node_attributes(G, 'hiv_status', hiv_status)

    return G

def TreatGon(G):
    gon_age= nx.get_node_attributes(G, 'gon_age')
    gon_status = nx.get_node_attributes(G, 'gon_status')
    gon_type =nx.get_node_attributes(G, 'gon_type')
    for nod in gon_age:
        age = gon_age[nod]
        if age == 2:
            gon_status[nod] = 0
            gon_age[nod] = 0
            gon_type[nod] = 0
    nx.set_node_attributes(G, 'gon_type', gon_type)
    nx.set_node_attributes(G,'gon_status', gon_status)
    nx.set_node_attributes(G,'gon_age',gon_age)
    return G

def calibrate_prep(G,p):

    #This function makes sure that the percent of people on prep remains constant
    amount_prep = math.floor(n*p)
    hiv_status = nx.get_node_attributes(G, 'hiv_status')
    current_prep = 0
    for nod in hiv_status:
        if hiv_status[nod] == 1:
            current_prep = current_prep +1

    dif = amount_prep - current_prep

    counter = 0
    for nod in hiv_status:
        if hiv_status[nod] == 2 and counter <=dif:
            hiv_status[nod] = 1
            counter = counter + 1

    n_prep = 0
    for nod in hiv_status:
        if hiv_status[nod] == 1:
            n_prep = n_prep + 1

    nx.set_node_attributes(G, 'hiv_status', hiv_status)

    return G

def MoveOneStep(G,p,st):

    ''' This function receives a graph as an input and advances one step, developing the following tasks:
    1 - Generates new edges between individuals that are not connected
    2 - Advances the stage of a relationship either making it steady or deleting it
    3 - Modify status in HIV status
    4- Modify status in Gonorrhea
    5- Updates amount of people under treatment or on Prep'''

    #Propagating diseases
    G = HIV_Spreading(G)
    G = Gonorrhea_Spreading(G)
    pr = 0.175 #percent of infected people getting on treatment
    G = TreatHIV(G,pr)
    G = TreatGon(G)
    G = calibrate_prep(G,p)
    G = AdvancingRelations(G)
    G = NewEdgesGeneration(G,st)
    return G

def GetMeasures(G):

    people_hiv = 0
    people_gon = 0
    hiv_status = nx.get_node_attributes(G,'hiv_status')
    gon_status = nx.get_node_attributes(G, 'gon_status')

    for i in hiv_status:
        if hiv_status[i] == 3 or hiv_status[i] == 4:
            people_hiv = people_hiv + 1

    for i in gon_status:
        if gon_status[i] == 1:
            people_gon = people_gon + 1

    return people_gon,people_hiv

def Deletion_Random(G,n_deletion):

    list_edges = []

    for edg in G.edges():
        list_edges.append(edg)

    for i in range(0,n_deletion):

        edge_todel = random.choice(list_edges)
        ini = edge_todel[0]
        fin = edge_todel[1]
        G.remove_edge(ini, fin)
        list_edges.remove(edge_todel)

    return G

def Minimizing_Weight(G,k):

    vih_status = nx.get_node_attributes(G,'hiv_status')
    # set of infected nodes
    infected = []
    susceptible = []
    for nod in vih_status:
        if vih_status[nod] ==4:
            infected.append(nod)
        elif vih_status[nod] == 1 or vih_status[nod] == 2:
            susceptible.append(nod)
    # k = a budget

    M = 20 #Number of samples
    A = G.edges()
    N = G.nodes()

    L = []
    n_paths = 0
    set_origins = []
    set_fin =[]
    count = 0
    for i in range(0,M):
        origin = random.choice(infected)
        set_origins.append(origin)
        fin = random.choice(susceptible)
        set_fin.append(fin)
        count = count + 1
        paths = nx.all_simple_paths(G, source=origin, target=fin)
        n_paths = n_paths + len(list(paths))
    while len(L)<= k:
        p_ideal = 0
        count = 0
        for edg in G.edges():
            count = count + 1
            G_copy = G
            ini = edg[0]
            tar = edg[1]
            G_copy.remove_edge(ini,tar)
            n_path = 0
            for i in range(0, M):
                origin = set_origins[i]
                fin = set_fin[i]
                paths = nx.all_simple_paths(G_copy, source=origin, target=fin)
                n_path = n_path + len(list(paths))
            dif = n_paths - n_path
            if dif > p_ideal:
                p_ideal = dif
                best_edge = edg
        L.append(edg)

    for edg in L:
        ini = edg[0]
        tar = edg[1]
        G.remove_edge(ini, tar)

    return G

def Edge_centrality(G,k):
    hiv_status = nx.get_node_attributes(G, 'hiv_status')
    susceptible = []
    infected =[]
    for i in hiv_status:
        if hiv_status[i] == 1 or hiv_status[i] == 2:
            susceptible.append(i)
        if hiv_status[i] == 4:
            infected.append(i)
    M = 10 #Sample size
    list_edges = []

    for edg in G.edges():
        list_edges.append(edg)

    dict_centrality = {}
    num_edges = 0
    while num_edges < M:
        edg = random.choice(list_edges)
        ini = edg[0]
        fin = edg[1]

        central = 0

        for infec in infected:
            for suscep in susceptible:
                if nx.has_path(G,infec,suscep):
                    min_paths = nx.all_shortest_paths(G,source=infec,target=suscep)
                    total_path = len(list(min_paths))
                    passing = 0
                    for path in min_paths:
                        if ini in path and fin in path:
                            passing = passing + 1
                    central = central + passing/total_path
        dict_centrality.update({edg:central})
        num_edges = +1
    sorted_dict = sorted(dict_centrality.items(), key=operator.itemgetter(1))

    edges_sorted = sorted_dict.keys()

    for edg in edges_sorted:
        ini = edg[0]
        fin = edg[1]
        G.remove_edge(ini, fin)

    return G

def classify_edges(G):
    infected =[]
    hiv_status  = nx.get_node_attributes(G, 'hiv_status')
    for edg in G.edges():
        ini = edg[0]
        fin = edg[1]
        if hiv_status[ini] == 4 or hiv_status[fin] == 4:
            infected.append(edg)
    infected = set(infected)
    return infected

def between_deletion(G,k):
    bet_dict = nx.edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None)
    infected = classify_edges(G)

    final_dict = {}

    for val in infected:
        final_dict.update({val: bet_dict[val]})

    sorted_dict = sorted(final_dict.items(), key=operator.itemgetter(1))
    counter = 0
    iter = 1
    print len(sorted_dict)
    while counter <= k:
        val = sorted_dict[-iter]
        edg = val[0]
        ini = edg[0]
        fin = edg[1]

        if G.has_edge(ini, fin):
            G.remove_edge(ini, fin)
            counter = counter + 1

        iter = iter + 1

    return G

def communicativity(G,k):

    G.remove_edges_from(G.selfloop_edges())
    com     = nx.communicability_centrality(G)
    counter = 0

    bet_dict = {}
    for edg in G.edges():
        ini = edg[0]
        fin = edg[1]
        value = com[ini]+com[fin]
        bet_dict.update({edg:value})

    infected = classify_edges(G)

    final_dict = {}

    for val in infected:
        final_dict.update({val:bet_dict[val]})

    sorted_dict = sorted(final_dict.items(), key=operator.itemgetter(1))
    counter = 0
    iter = 1
    print len(sorted_dict)
    while counter <= k:
        val = sorted_dict[-iter]
        edg = val[0]
        ini = edg[0]
        fin = edg[1]

        if G.has_edge(ini,fin):
            G.remove_edge(ini, fin)
            counter = counter + 1

        iter = iter +1

    return G

def implement_deletion(G,index,k):
    if index == 1:
        G = Deletion_Random(G,k)
    elif index == 2:
        G = Edge_centrality(G,k)
    elif index ==3:
        G = Minimizing_Weight(G,k)
    elif index ==4:
        G = between_deletion(G,k)
    elif index ==5:
        G= communicativity(G,k)
    return G

def measure_gonorrhea(G):
    gon_type = nx.get_node_attributes(G, 'gon_type')
    infected_prep = 0
    infected_notprep = 0
    for nod in G.nodes():
        if gon_type[nod] == 1:
            infected_prep = infected_prep + 1
        elif gon_type[nod] == 2:
            infected_notprep = infected_notprep + 1

    infected_total = infected_notprep + infected_prep
    return infected_prep, infected_notprep, infected_total

def measure_type(G):
    casual = 0
    steady = 0
    hiv_type = nx.get_node_attributes(G, 'hiv_type')
    for nod in G.nodes():
        if hiv_type[nod] == 1:
            casual = casual +1
        elif hiv_type[nod] == 2:
            steady = steady + 1
    return casual,steady

def figure1(G,p,n_steps):

    cum_prep = []
    cum_total = []
    cum_noprep = []

    prep_val = 0
    total_val = 0
    noprep_val = 0
    steps = range(0,n_steps + 1)
    count = 0

    prep_val, noprep_val, total_val = measure_gonorrhea(G)

    cum_prep.append(prep_val)
    cum_total.append(total_val)
    cum_noprep.append(noprep_val)
    st = 0.3

    while count < n_steps:
        G = MoveOneStep(G,p,st)
        prep_val, noprep_val,total_val = measure_gonorrhea(G)
        cum_prep.append(prep_val)
        cum_total.append(total_val)
        cum_noprep.append(noprep_val)
        count = count + 1
        print count

    cumul_prep = []
    cumul_total = []
    cumul_noprep = []

    for i in range(0,len(cum_prep)):
        sum_prep = 0
        sum_noprep = 0
        sum_total = 0
        for j in range(0,i):
            sum_prep = sum_prep + cum_prep[j]
            sum_noprep = sum_noprep + cum_noprep[j]
            sum_total = sum_total + cum_prep[j] + cum_noprep[j]

        cumul_prep.append(sum_prep)
        cumul_total.append(sum_total)
        cumul_noprep.append(sum_noprep)



    #Cumulative values
    plt.hold(True)
    plt.plot(steps[1:],cumul_noprep[1:],label ='Prep')
    plt.plot(steps[1:],cumul_prep[1:], label='No Prep')
    plt.plot(steps[1:], cumul_total[1:], label='Totals')
    plt.hold(False)
    plt.legend()
    plt.xlabel('Semesters')
    plt.ylabel('Number of people infected')
    plt.savefig('Gonorrhea.eps', format='eps')


def figure2(G,p,n):
    steady = []
    casual = []
    st = 0.3
    steady_val = 0
    casual_val = 0
    steps = range(0, n_steps + 1)

    casual_val,steady_val = measure_type(G)

    steady.append(steady_val)
    casual.append(casual_val)
    count = 0

    while count < n_steps:
        G = MoveOneStep(G, p,st)
        casual_val, steady_val = measure_type(G)
        casual.append(casual_val)
        steady.append(steady_val)
        count = count + 1
        print count

    plt.hold(True)
    plt.plot(steps[1:], casual[1:], label='Casual')
    plt.plot(steps[1:], steady[1:], label='Steady')

    plt.hold(False)
    plt.legend()
    plt.xlabel('Semesters')
    plt.ylabel('Number of people infected')
    plt.savefig('RelType.eps', format='eps')

def figure3(G,p,n_steps,k):

    steps = range(0, n_steps + 1)
    people_gon_or, people_hiv_or = GetMeasures(G)
    st = 0.3
    G_r = G
    gon_status_r =[]
    hiv_status_r = []
    gon_status_r.append(people_gon_or)
    hiv_status_r.append(people_hiv_or)

    counter =1
    #Random Deletion
    print'Random'
    while counter <= n_steps:

        G_r = MoveOneStep(G_r, p,st)
        index = 1
        G_r = implement_deletion(G_r, index, k)
        people_gon, people_hiv = GetMeasures(G_r)
        gon_status_r.append(people_gon)
        hiv_status_r.append(people_hiv)
        counter = counter + 1
        print counter

    #Betweenness

    G_b = G
    gon_status_b = []
    hiv_status_b = []
    gon_status_b.append(people_gon_or)
    hiv_status_b.append(people_hiv_or)

    counter = 1
    # Betweenees Deletion
    print 'Betweeness'
    while counter <= n_steps:

        G_b = MoveOneStep(G_b, p,st)
        index = 4
        G_b = implement_deletion(G_b, index, k)
        people_gon, people_hiv = GetMeasures(G_b)
        gon_status_b.append(people_gon)
        hiv_status_b.append(people_hiv)
        counter = counter + 1
        print counter

    #Connectivity

    G_c = G
    gon_status_c = []
    hiv_status_c = []
    gon_status_c.append(people_gon_or)
    hiv_status_c.append(people_hiv_or)

    counter = 1
    # Connectivity
    print 'Connec'
    while counter <= n_steps:
        G_c = MoveOneStep(G_c, p,st)
        index = 5
        G_c = implement_deletion(G_c, index, k)
        people_gon, people_hiv = GetMeasures(G_c)
        gon_status_c.append(people_gon)
        hiv_status_c.append(people_hiv)
        counter = counter + 1
        print counter

    plt.plot(True)
    div_r = hiv_status_r[1]
    div_b = hiv_status_b[1]
    div_c = hiv_status_c[1]

    for i in range(1,len(hiv_status_r)):
        hiv_status_r[i] = 1.0*hiv_status_r[i]/div_r
        hiv_status_b[i] = 1.0 * hiv_status_b[i] / div_b
        hiv_status_c[i] = 1.0 * hiv_status_c[i] / div_c


    plt.plot(steps[1:],hiv_status_r[1:],label='Random')
    plt.plot(steps[1:],hiv_status_b[1:],label='Infected - Susceptibility Betweennness')
    plt.plot(steps[1:], hiv_status_c[1:], label='Minimization of Connectivity')
    plt.xlabel('Semesters')
    plt.ylabel('Evolution of initial infected people')
    plt.legend()
    plt.show()
    plt.savefig('Deletion.eps', format='eps')


def figure4(G,n_steps):
    pro = [0,0.2,0.4,0.6,0.8]
    hiv_vec = []
    gon_vec = []
    steps = range(0, n_steps + 1)

    hiv_val,gon_val = GetMeasures(G)
    hiv_vec.append(hiv_val)
    gon_vec.append(gon_val)

    for idx in range(0,len(pro)):
        norm = 1
        val = pro[idx]
        st = 0.3
        GraphGeneration(n, val,st)
        for i in range(0,n_steps):
            G = MoveOneStep(G,val,st)
            gon,hiv = GetMeasures(G)
            hiv_vec.append(hiv)
            if i == 0:
                gon_vec.append(gon)
            elif i> 0:
                gon_vec.append(gon+gon_vec[i-1])
            if i == 1:
                norm = hiv_vec[i]
        for k in range(0,len(hiv_vec)):
            hiv_vec[k] = 1.0*hiv_vec[k]/norm

        plt.hold(True)
        plt.plot(steps[1:],hiv_vec[1:],label = str(val))
        plt.legend()

        hiv_vec =[]
        gon_vec = []
        hiv_vec.append(hiv_val)
        gon_vec.append(gon_val)
    plt.legend()
    plt.xlabel('Semesters')
    plt.ylabel('Infected people as proportion of original people infected')
    plt.savefig('p.eps', format='eps')
    plt.show()

def figure5(G):
    pro = [0, 0.2, 0.4, 0.6, 0.8]
    hiv_vec = []
    gon_vec = []
    steps = range(0, n_steps + 1)

    hiv_val, gon_val = GetMeasures(G)
    hiv_vec.append(hiv_val)
    gon_vec.append(gon_val)

    for idx in range(0, len(pro)):
        norm = 1
        val = 0.35
        st = pro[idx]
        GraphGeneration(n, val, st)
        for i in range(0, n_steps):
            G = MoveOneStep(G, val, st)
            gon, hiv = GetMeasures(G)
            hiv_vec.append(hiv)
            if i == 0:
                gon_vec.append(gon)
            elif i > 0:
                gon_vec.append(gon + gon_vec[i - 1])
            if i == 1:
                norm = hiv_vec[i]
        for k in range(0, len(hiv_vec)):
            hiv_vec[k] = 1.0 * hiv_vec[k] / norm

        plt.hold(True)
        plt.plot(steps[1:], hiv_vec[1:], label=str(st))
        plt.legend()

        hiv_vec = []
        gon_vec = []
        hiv_vec.append(hiv_val)
        gon_vec.append(gon_val)
    plt.legend()
    plt.xlabel('Semesters')
    plt.ylabel('Infected people as proportion of original people infected')
    plt.savefig('p.eps', format='eps')
    plt.show()

def measure_both(G):
    hiv_only = 0
    gon_only = 0
    both = 0
    hiv_status = nx.get_node_attributes(G, 'hiv_status')
    gon_status = nx.get_node_attributes(G, 'gon_status')

    for nod in G.nodes():
        if hiv_status[nod] > 2 and gon_status > 0:
            both = both +1

        elif hiv_status[nod] < 3 and gon_status > 0:
            gon_only = gon_only + 1
            hiv_only = hiv_only + 0.4
    print 'Hey'
    print hiv_only
    return hiv_only,gon_only,both

def figure6(G,p):
    only_hiv = []
    only_gon = []
    both =[]
    steps = range(0, n_steps + 1)
    count = 0

    hiv_val, gon_val, both_val = measure_both(G)

    only_hiv.append(hiv_val)
    only_gon.append(gon_val)
    both.append(both_val)
    st = 0.3

    while count < n_steps:
        G = MoveOneStep(G, p, st)
        hiv_val, gon_val, both_val = measure_both(G)
        only_hiv.append(hiv_val)
        only_gon.append(gon_val)
        both.append(both_val)
        count = count + 1
        print count

    # Cumulative values
    plt.hold(True)
    plt.plot(steps[1:], only_hiv[1:], label='Only HIV')
    plt.plot(steps[1:], only_gon[1:], label='Only Gon')
    plt.plot(steps[1:], both[1:], label='both')
    plt.hold(False)
    plt.legend()
    plt.xlabel('Semesters')
    plt.ylabel('Number of people infected')
    plt.show()
    plt.savefig('deag.eps', format='eps')



if __name__ == "__main__":
    n = 1000
    p = 0.35
    st = 0.3
    [G,p] = GraphGeneration(n,p,st)
    n_steps = 5

    counter = 1
    gon_status =[]
    hiv_status =[]
    k = 10
    #Figure 1: gonorrhea cumulative, figure 2: steady and casual, figure 3: edge deletion effect,\
    #Figure 4: sensitibity analysis for prep, figure 5  =, sensitivity analysis for casual and steady
    #Figure 6: combination of hiv and gonorrhea or hiv

    figure1(G,p,n_steps)
    #figure2(G,p,n_steps)
    #figure3(G,p,n_steps,k)
    #figure4(G,n_steps)
    #figure5(G,p)
    #figure6(G,p)

    print 'Done with analysis'