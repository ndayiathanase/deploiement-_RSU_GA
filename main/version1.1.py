import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame as df
import pickle

import tkinter as tk
# from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox
from tkinter import filedialog

nodes,edges2,rayon,  size, distance_matrix, folder_selected, tot_sum_edges = None, None, None, None, None, None, None

#%% Tools
def distance_deux_points(x1,y1,x2,y2):
    return np.sqrt( np.power((x2-x1),2) + np.power((y1-y2),2) )

def get_node_index(key):
    global nodes, edges2
    
    node_id = nodes.node_id
    index = node_id[node_id==key].index[0]
    return index

def node_to_coord(i):
    global nodes, edges2
    pp3 = nodes.iloc[i]
    return [pp3.node_x, pp3.node_y]

def from_edges_to_nodes(i):
    global nodes, edges2
    edg = edges2.iloc[i]
    node_orig = get_node_index(edg.edge_from)
    node_extr =  get_node_index(edg.edge_to)
  
    return [node_orig, node_extr]

def nodes_coverage(liste,rayon)  :
  nodes_covered = dict()
  for i in liste:
    nodes_covered[i] = list()
    # nod = [element for element in list(range(size)) if element != i]
    for j in range(size):
      if distance_matrix[i,j] < rayon:
        nodes_covered[i].append(j)

  return nodes_covered

def distance_ligne_point(p1,p2,p3):
  p1=np.array(p1)
  p2=np.array(p2)
  p3=np.array(p3)
  return abs ( np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1) )

import math
import portion

def coverage_orig_in(node_1,node_2,p3c,rayon):
  p1 = node_to_coord(node_1)
  p2 = node_to_coord(node_2)
  p3c = node_to_coord(p3c)
  oO = distance_ligne_point(p1,p2,p3c)

  oB = distance_deux_points(p3c[0],p3c[1] , p2[0] , p2[1])
  oA = distance_deux_points(p3c[0],p3c[1], p1[0] , p1[1])
  AB = distance_deux_points(p1[0] , p1[1],p2[0] , p2[1])

  OA = math.sqrt( math.pow(oA,2) - math.pow(oO,2))
  OB = math.sqrt( math.pow(oB,2) - math.pow(oO,2))

  Oc = math.sqrt( math.pow(rayon,2) - math.pow(oO,2))

  if AB > OA and AB > OB:
      return portion.closed(0, Oc + OA)
  else :
      return portion.closed(0, Oc - OA)

def coverage_orig_out(node_1,node_2,p3c,rayon):
  p1 = node_to_coord(node_1)
  p2 = node_to_coord(node_2)
  p3c = node_to_coord(p3c)
  oO = distance_ligne_point(p1,p2,p3c)

  # oB = distance_deux_points(p3c[0],p3c[1] , p2[0] , p2[1])
  oA = distance_deux_points(p3c[0],p3c[1], p1[0] , p1[1])
  AB = distance_deux_points(p1[0] , p1[1],p2[0] , p2[1])

  OA = math.sqrt( math.pow(oA,2) - math.pow(oO,2))
  # OB = math.sqrt( math.pow(oB,2) - math.pow(oO,2))
  
  Oc = math.sqrt( math.pow(rayon,2) - math.pow(oO,2))
  
  AC = OA - Oc
  return portion.closed(AC, AB)


def coverage_tot_out(p1,p2,p3c,rayon):
  p1 = node_to_coord(p1)
  p2 = node_to_coord(p2)
  p3c = node_to_coord(p3c)
  oO = distance_ligne_point(p1,p2,p3c)

  oA = distance_deux_points(p3c[0],p3c[1], p1[0] , p1[1])
  OA = math.sqrt( math.pow(oA,2) - math.pow(oO,2))
  AB = distance_deux_points(p1[0] , p1[1],p2[0] , p2[1])

  Oc = math.sqrt( math.pow(rayon,2) - math.pow(oO,2))
  Ac = OA-Oc
  AC = Ac + (Oc * 2)
  
  oB = distance_deux_points(p3c[0],p3c[1] , p2[0] , p2[1])
  OB = math.sqrt( math.pow(oB,2) - math.pow(oO,2))
  
  if oO < rayon :
      if AB > OA and AB > OB:
          return  portion.closed(Ac,AC)
      else :
          return portion.empty()
      
  else:
      return portion.empty()

def matrix_edges_nodes() :
  
  distances_edges_nodes = np.zeros((size,edges2.shape[0]))
  for i in range(size):
    print('---- Ligne :',i,'...')
    for j in range(edges2.shape[0] ):
      pp3 = nodes.iloc[i]
      p3 = [pp3.node_x, pp3.node_y]

      orig = from_edges_to_nodes(j)[0]
      p1 = node_to_coord(orig)

      extr =  from_edges_to_nodes(j)[1]
      p2 = node_to_coord(extr)

      # print(i,pp1,pp2)
      # print(p1,p2,p3)
      distances_edges_nodes[i,j] =  distance_ligne_point(p1,p2,p3) 
    print('Ligne :',i, distances_edges_nodes[i])
  print(df(distances_edges_nodes).head())
  return distances_edges_nodes

def update_cov(new, main):
  for i in range(len(new)):
    if main[i] < new[i]:
      main[i] = new[i]
  return main

matrix_edge_node_data = None

def coverage_of_all(rayon):
  liste = list(range(size))
  size_edge = edges2.shape[0]
  covs = list( range(size ))

  covered_nodes = nodes_coverage(liste,rayon)
  dist = df (  matrix_edge_node_data  )
  binaire =  dist < rayon
  for i in liste:
    # print('---------')
    temp_dict = dict()
    for j in range( size_edge ) :

      if binaire.iloc[i,j]== True :
        inf= float('inf')
        temp = portion.empty()
        A = from_edges_to_nodes(j)[0]
        B = from_edges_to_nodes(j)[1]
        cov = 0

        if A in covered_nodes[i] and B not in covered_nodes[i]:
          # print('1',i,A,B)
          cov = coverage_orig_in(A,B, i, rayon)
          # print(distance_matrix[A,B] , cov,0,0, '  --- ', A, 'dedans', B, 'dehors')
            
          temp =temp|cov

        if A not in covered_nodes[i] and B in covered_nodes[i]:
          # print('2',i,A,B)
          cov = coverage_orig_out(A,B, i, rayon)
          # print(distance_matrix[A,B], 0,0,cov, '  --- ', A, 'dehors', B, 'dedans')
          temp = temp|cov
         
        if A in covered_nodes[i] and B in covered_nodes[i]:
          cov = portion.closed(0, distance_matrix[A,B] )
          temp = temp|cov
          
        if A not in covered_nodes[i] and B not in covered_nodes[i]:
          cov = portion.empty()
          # print(distance_matrix[A,B] , 0,cov, 0, '  --- ', A, B, ' dedans ou out')
          temp = temp|cov

        if cov != portion.empty() :
          temp_dict[j] = [temp,(A,B,distance_matrix[A,B])]
          print ('node ',i,'edge ',j, 'valeurs ',[temp,(A,B,distance_matrix[A,B])])
      covs[i] = temp_dict
    print('\n\nLigne ',i, ' : ',temp_dict )
    # print(i, temp_dict)
  return covs
#%% Op
def op_open_files():
    if folder_selected == None :
        messagebox.showwarning(title='Attention', message='Dossier non selectionnee') 
        return   
    global nodes, edges2, size
    nodes = pd.read_csv(folder_selected+'/plain.nod.csv',sep=';')
    edges = pd.read_csv(folder_selected+'/plain.edg.csv', on_bad_lines='skip', sep=';')
    
    nodes = nodes[['node_id','node_type','node_x', 'node_y']]
    print ("***** Taille of Node Data *****\n \t\t", nodes.shape)
    
    nodes.dropna(inplace=True)
    nodes.reset_index(inplace=True,drop=True)
    
    edges = edges[['edge_from','edge_to', 'edge_id' ]]
    edges.columns
    
    edges2 = pd.DataFrame(columns = edges.columns)
    for index, row in edges.iterrows():
        frm =  row['edge_from']
        to = row['edge_to']
        
        replic = edges2.loc[(edges2['edge_from']==to) & (edges2["edge_to"]==frm)].shape[0]
        if (replic == 0):
          edges2.loc[len(edges2.index)] = row
          
    edges2.dropna(inplace=True)
    size = nodes.shape[0]
    print ("***** Taille of Edges Data *****\n \t\t", edges2.shape)
    
    plot_map([0],0)
    
def op_create_distance_matrix():
    global distance_matrix,nodes
    print ("\n\n***** DISTANCE MATRIX *****\n \t\t")
    
    distance_matrix = np.zeros((size,size))
    
    if messagebox.askyesno("Est ce que DM existe?")==True:
        with open(folder_selected + '/distance_matrix.pkl' , 'rb') as f:
            distance_matrix = pickle.load(f)  
    else :
        print('\n\n********** Calcul de la Distance Matrix **********')
        for i in range(size):
          for j in range(size):
    
            distance_matrix[i,j] = distance_deux_points(nodes.iloc[i].node_x, nodes.iloc[i].node_y, nodes.iloc[j].node_x, nodes.iloc[j].node_y) 
          print('distance pour ',i,'row est calculee')
        
        with open(folder_selected + '/distance_matrix.pkl' , 'wb') as f:
            pickle.dump(distance_matrix,f) 
            print('\n\n********** Distance Matrix is saved **********') 
    print(df(distance_matrix).head())
        
def long_edge(seri) :
    
    orig = get_node_index(seri['edge_from'])
    extr = get_node_index(seri['edge_to'])

    p1 = node_to_coord(orig)
    p2 = node_to_coord(extr)
    d = distance_deux_points(p1[0],p1[1],p2[0], p2[1])
    print('Distance ',orig,' to ',extr,' = ', d )

    return d

def op_add_long_edges():
    print('\n\n********** Creation de la colonne Longueur dans Edge Matrix **********') 

    global edges2, tot_sum_edges
    edges2['edge_longueur']= edges2.apply(long_edge,axis=1)
    tot_sum_edges = edges2['edge_longueur'].sum()
    print(df(edges2).head())
    
adjacence_matrix = None
def op_adj_matrix():
 
    global size,edges2, adjacence_matrix
    t = messagebox.askyesno("Adj Matrix Existe ?")
    print(t)

    if t ==True:

        with open(folder_selected + '/adjacence_matrix.pkl' , 'rb') as f:
            adjacence_matrix = pickle.load(f)
        with open(folder_selected + '/adjacence_matrix_edge.pkl' , 'rb') as f:
            adjacence_matrix_edge = pickle.load(f)
    else:
        print('\n\n********** Creation de ADJ Matrix ... **********')
        adjacence_matrix = np.zeros((size,size))
        adjacence_matrix_edge = np.zeros((size,size),dtype=float)
        for ind,row in edges2.iterrows():
          i = get_node_index(row.edge_from)
          j =  get_node_index(row.edge_to)
          adjacence_matrix[i,j] = 1
          adjacence_matrix_edge[i,j]  = ind+1
          
          with open(folder_selected + '/adjacence_matrix.pkl' , 'wb') as f :
              pickle.dump(adjacence_matrix,f)
          with open(folder_selected + '/adjacence_matrix_edge.pkl' , 'wb') as f :
              pickle.dump(adjacence_matrix_edge,f)
    print('\n\n********** ADJ Matrix **********')
    print ( df(adjacence_matrix) )

def op_edg_node_matrix():
    global matrix_edge_node_data
    answer = messagebox.askyesno("Edges-Nodes Distance Matrix existe ?")
    if ( answer == True):
        with open(folder_selected + '/matrix_edges_nodes.pkl', 'rb') as f:
            matrix_edge_node_data = pickle.load(f)
    else :
        print('\n\n********** Creation de Edges-Nodes Distance Matrix ... **********')
        matrix_edge_node_data = matrix_edges_nodes() 
        with open(folder_selected + '/matrix_edges_nodes.pkl', 'wb') as f:
            pickle.dump(matrix_edge_node_data, f)
    print('\n\n********** Edges-Nodes Distance Matrix **********')
    print ( df(matrix_edge_node_data).head() )
    
aa = None
def op_coverage_of_all():
    global aa
    global rayon
    print('********Coverage Data********************')
    t = messagebox.askyesno("COVERAGE DATA existe?")
    if t ==True :
        with open(folder_selected + '/data_coverage.pkl', 'rb') as f:
          aa = pickle.load(f)
    else :  
        print('************ Calcul de zones de couverture ....')
        aa = coverage_of_all(rayon)
        with open(folder_selected + '/data_coverage.pkl', 'wb') as f:
            pickle.dump(aa,f)

    print(aa[:5])
    print('\n\n\n\n******************** GA est pret ! *******************')
#%%
import threading
def run_ga():

        def click_button():
            multi_ga2()

        threading.Thread(target=click_button).start()
test2 = None
def run_ga2():
    global test2

    num_rsu =  int ( txt_num_rsu.get() )
    gen_size = int ( txt_gen_size.get() )
    reprod_size = int (txt_reprod_size.get() )
    num_iter = int (txt_num_iter.get() )
    print(num_rsu, gen_size, reprod_size,num_iter)
    test2 = GeneticAlgos(tot_sum_edges, aa, nodes, edges2, num_rsu ,  gen_size,      reprod_size,           num_iter ,        False, False)
    test2.optimize()
def multi_ga2( ) :
    global test2
    num_rsu =  int ( txt_num_rsu.get() )
    gen_size = int ( txt_gen_size.get() )
    reprod_size = int (txt_reprod_size.get() )
    num_iter = int (txt_num_iter.get() )
    results= []
    
    print(num_rsu, gen_size, reprod_size,num_iter)
    test2 = GeneticAlgos(tot_sum_edges, aa, nodes, edges2, num_rsu ,  gen_size,      reprod_size,           num_iter ,        False, False)
    test2.optimize()
    
    while True:
        num_rsu = num_rsu + 1

        print(num_rsu, gen_size, reprod_size,num_iter)
        test2 = GeneticAlgos(tot_sum_edges, aa, nodes, edges2, num_rsu ,  gen_size,      reprod_size,           num_iter ,        False, False)
        test2.optimize() 
        
        results.append(test2.top_chromosome)
        pourcent = 100 - ( test2.top_chromosome.fitness *100 / test2.tot )
        print('size :',size, 'num rsu :',num_rsu,'tot :', test2.tot, 'pourcent :',pourcent)
        print('################# NEXT STEP :',num_rsu,'RSU ###################')
        if pourcent > 95 :
            break
        if num_rsu >= size-1:
            break

    print (results)
    print ('Couverture Total :', test2.tot)
    return results
def print_to_text(text,msg):
    text.insert(tk.END, msg)
    text.see(tk.END)
    
j = 0    
def test():
    global j
    t = ''
    
    for i in range(10):
        j = j+1
        t = t + str(j)+ 'abc\n'
    txt_edit.insert(tk.END, t)
    txt_edit.see(tk.END)
    
def open_file():
    global folder_selected
 
    folder_selected = filedialog.askdirectory()
    messagebox.showinfo(title='Infos', message= folder_selected) 
    # """Open a file for editing."""
    # filepath = askopenfilename(
    #     filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    # )
    # if not filepath:
    #     return
    # txt_edit.delete("1.0", tk.END)
    # with open(filepath, mode="r", encoding="utf-8") as input_file:
    #     text = input_file.read()
    #     txt_edit.insert(tk.END, text)
    # window.title(f"Simple Text Editor - {filepath}")
def save_file():
    global rayon
    print(test2.top_chromosome.content)
    plot_map(test2.top_chromosome.content,rayon)
    
def save_file2():
    global rayon
    #print(test2.top_chromosome.content)
    plot_map(list(range(size)),rayon)    
# def save_file_unused():
    # """Save the current file as a new file."""
    # filepath = asksaveasfilename(
    #     defaultextension=".txt",
    #     filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
    # )
    # if not filepath:
    #     return
    # with open(filepath, mode="w", encoding="utf-8") as output_file:
    #     text = txt_edit.get("1.0", tk.END)
    #     output_file.write(text)
    # window.title(f"Simple Text Editor - {filepath}")

window = tk.Tk()
window.title("Deployment RSU")

window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

# txt_edit = tk.Text(window)
frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
frm_buttons.pack( side = tk.LEFT )

frm_outputs = tk.Frame(window, relief=tk.RAISED, bd=2)
frm_outputs. pack( side = tk.RIGHT )

import tkinter.scrolledtext as tkscrolled

#https://stackoverflow.com/questions/13832720/how-to-attach-a-scrollbar-to-a-text-widget
default_text = '1234'
width, height = 90,30
txt_edit = tkscrolled.ScrolledText(frm_outputs, width=width, height=height, wrap='word')
txt_edit.insert(1.0, default_text)
txt_edit.grid(row=0, column=0)

txt_edit = tkscrolled.ScrolledText(frm_outputs, width=width, height=height, wrap='word')
txt_edit.insert(1.0, default_text)
txt_edit.grid(row=0, column=0)

def start():
        
        def click_button():
            global rayon
            rayon =  int ( txt_rayon_rsu.get() )
            op_open_files()
            op_create_distance_matrix() 
            op_add_long_edges()
            op_adj_matrix()
            op_edg_node_matrix()
            op_coverage_of_all()

        threading.Thread(target=click_button).start()
        
lbl_num_rsu = tk.Label(frm_buttons, text="Nombre de RSU")
lbl_gen_size = tk.Label(frm_buttons, text="Taille Generation")
lbl_reprod_size = tk.Label(frm_buttons, text="Taille Reproduction")
lbl_num_iter = tk.Label(frm_buttons, text="Nombre Iterations")

btn_open = tk.Button(frm_buttons, text="Select Fichiers", command=open_file)
btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

btn_t = tk.Button(frm_buttons, text='Preparer Data', command = start)
btn_t.grid(row=2, column=0, sticky='ew', padx=5, pady=5)

btn_run = tk.Button(frm_buttons, text='Run GA', command = run_ga)
btn_run.grid(row=8, column=0, sticky='ew', padx=5, pady=5)

btn_save = tk.Button(frm_buttons, text="Map", command=save_file)
btn_save.grid(row=9, column=0, sticky="ew", padx=5, pady=5)


lbl = tk.Label(frm_buttons, text=" -Parametres de GA- ")
lbl.grid(row=3,column = 0, sticky="ew", padx=5, pady=5)

txt_rayon_rsu = tk.Entry(frm_buttons,width=10)
txt_rayon_rsu.grid(row=1,column = 1,sticky='ew', padx=5)
lbl_rayon = tk.Label(frm_buttons, text="Rayon")
lbl_rayon.grid(row=1,column = 0, sticky="ew", padx=5)

lbl_num_rsu.grid(row=4,column = 0)
lbl_gen_size.grid(row=5,column = 0)
lbl_reprod_size.grid(row=6,column = 0)
lbl_num_iter.grid(row=7,column = 0)

txt_num_rsu = tk.Entry(frm_buttons,width=10)
txt_gen_size = tk.Entry(frm_buttons,width=10)
txt_reprod_size = tk.Entry(frm_buttons,width=10)
txt_num_iter = tk.Entry(frm_buttons,width=10)

txt_num_rsu.grid(row=4,column = 1,sticky='ew', padx=5)
txt_gen_size.grid(row=5,column = 1,sticky='ew', padx=5)
txt_reprod_size.grid(row=6,column = 1,sticky='ew', padx=5)
txt_num_iter.grid(row=7,column = 1,sticky='ew', padx=5)


def redirector(inputStr):
    
    txt_edit.insert(tk.END, inputStr)
    txt_edit.see(tk.END)
    # txt_edit.insert(INSERT, inputStr)

sys.stdout.write = redirector
        

#%%


#%%  plot

plt.rcParams["figure.figsize"] = [17, 7]
plt.rcParams["figure.autolayout"] = True

def plot_map(chromosome, rayon):
  theta = np.linspace(0, 2*np.pi, 100)
  r = rayon

  for ind, row in edges2.iterrows():

    orig = get_node_index(row.edge_from)
    extr = get_node_index(row.edge_to)
    if ind==12:
      print(orig,"--",extr)


    orig_x = nodes.iloc[orig].node_x
    orig_y = nodes.iloc[orig].node_y

    extr_x = nodes.iloc[extr].node_x
    extr_y = nodes.iloc[extr].node_y

    point1 = [orig_x, orig_y]
    point2 = [extr_x, extr_y]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values, 'bo', linestyle="--")

    plt.text(point1[0]-0.015, point1[1]+0.25, orig)
    plt.text(point2[0]-0.050, point2[1]-0.25, extr)

    if (orig in chromosome) :

      x1 = ( r*np.cos(theta) ) + orig_x
      x2 = (r*np.sin(theta) ) + orig_y
      plt.plot(x1, x2)
      
    if (extr in chromosome) : 
      x1 = ( r*np.cos(theta) ) + extr_x
      x2 = (r*np.sin(theta) ) + extr_y
      plt.plot(x1, x2)

  plt.show()
  

#%%

def evaluation (edge, temp):
  longueur = edges2.iloc[edge,3] 
  reste_sauf_extremites = longueur - temp[0] - temp[2]
  if reste_sauf_extremites < 0 :
    return 0
  else :
    reste = reste_sauf_extremites - ( temp[1]*0.5 )
    if reste < 0:
      return 0
    else :
      return reste


#%%

import time
class Chromosome:
    
    """
    Class Chromosome represents one chromosome which consists of genetic code and value of 
    fitness function.
    Genetic code represents potential solution to problem - the list of locations that are selected
    as medians.
    """
    
    def __init__(self, content, fitness):
        self.content = content
        self.fitness = fitness
    def __str__(self): return "%s f=%d" % (self.content, self.fitness)
    def __repr__(self): return "%s f=%d" % (self.content, self.fitness)
    

#%%

import random

class GeneticAlgos:
   
    def __init__(self, tot, all_cov, nodes, edges, num_rsu, gen_size, reprod_size, num_iter, init_population_with_center_point=False, apply_hypermutation=False):
        
        self.tot = tot
        self.i = 0
        self.nodes = nodes
        self.edges2 = edges
        self.cover = all_cov

        self.num_intersections = len(nodes)
        self.num_edges = len(edges2)
        self.num_rsu = num_rsu
        
        self.init_population_with_center_point = init_population_with_center_point
        self.apply_hypermutation = apply_hypermutation
    
        self.iterations = num_iter                                      # Maximal number of iterations
        self.current_iteration = 0
        self.generation_size = gen_size                                 # Number of individuals in one generation
        self.reproduction_size = reprod_size                               # Number of individuals for reproduction
        
        self.mutation_prob = 0.3                               # Mutation probability
        self.hypermutation_prob = 0.03                            # Hypermutation probability
        self.hypermutation_population_percent = 10

        self.top_chromosome = None      # Chromosome that represents solution of optimization process
        # print ('init pop',   self.fitness([5, 1, 14, 20, 19],3)   )

    def mutation(self, chromosome):
      # print ( self.i, '-- mutation test',  self.fitness([5, 1, 14, 20, 19],3)   )
      self.i = self.i + 1
      """ 
      Applies mutation over chromosome with probability self.mutation_prob
      In this process, a randomly selected median is replaced with a randomly selected demand point.
      """
      choix = list(range(self.num_rsu))
      mp = random.random()
      if mp < self.mutation_prob:
          # index of randomly selected median:
          # i = random.randint(0, len(chromosome)-1)
          # demand_points = [element for element in range(0,self.num_intersections) if element not in chromosome] 
          # chromosome[i] = random.choice(demand_points)
          
          f = random.sample(choix , self.num_rsu//4 + 1)  

          for j in f :
              # demand points without current medians:
              demand_points = [element for element in range(0,self.num_intersections) if element not in chromosome] 
              # replace selected median with randomly selected demand point:
              chromosome[j] = random.choice(demand_points)
          
      return chromosome

    def crossover(self, parent1, parent2):
      # print (self.i, '-- cross over pop',   self.fitness([5, 1, 14, 20, 19],3)   )
      self.i = self.i + 1

      print ( 'cross over' , self.fitness([1,2,3,4],3)   )
      
      # print_to_text(txt_edit,'cross over***\n')
      identical_elements = [element for element in parent1 if element in parent2]
        
        # If the two parents are equal to each other, one of the parents is reproduced unaltered for the next generation 
        # and the other parent is deleted, to avoid that duplicate individuals be inserted into the population.
      if len(identical_elements) == len(parent1):
            return parent1, None
      
      child1 = []
      child2 = []

      exchange_vector_for_parent1 = [element for element in parent1 if element not in identical_elements]
      exchange_vector_for_parent2 = [element for element in parent2 if element not in identical_elements]   
        
      c = random.randint(0, len(exchange_vector_for_parent1)-1)
        
      for i in range(c):
        exchange_vector_for_parent1[i], exchange_vector_for_parent2[i] = exchange_vector_for_parent2[i], exchange_vector_for_parent1[i]

      child1 = identical_elements + exchange_vector_for_parent1
      child2 = identical_elements + exchange_vector_for_parent2
        
      return child1, child2

    def evaluate_length_portion(self, a):
        if a == portion.empty():
            return 0
        if a.atomic:
            return a.upper - a.lower
        else:
            l = 0
            for i in range(len(a)):
                t = a[i]
                l = l + (  t.upper - t.lower )
            return l
    def evaluate_portions(self ,chromosome):
        with open(folder_selected + '/data_coverage.pkl', 'rb') as f:
            aa = pickle.load(f)
        liste = [portion.empty() for i in range(len(edges2) )]
       
        for i in chromosome:
            cov_dict = aa[i]
            for key, value in  cov_dict.items():
                
                liste[key] = liste[key] | value[0]  
        return liste
    def evaluate_portion_list(self, liste) :
        score = 0
        for i in range ( len (liste) ):
            # print(liste[i])
            score = score + self.evaluate_length_portion(liste[i])
        return score
    
    def fitness(self,chromosome,k) :
        t = self.evaluate_portions(chromosome)
        score = self.evaluate_portion_list(t)

        return  self.tot - score
      # return 10

    def initial_random_population(self):
      # print (self.i, '-- random pop',   self.fitness([5, 1, 14, 20, 19],3)   )

      self.i = self.i + 1

      """ 
        Creates initial population by generating self.generation_size random individuals.
        Each individual is created by randomly choosing p facilities to be medians.
        """
      init_population = []
      for k in range(self.generation_size):
        rand_intersections = []
        n = list(range(self.num_intersections))
        for i in range(self.num_rsu):
          rand_node = random.choice(n)
          rand_intersections.append(rand_node)
          n.remove(rand_node)
        init_population.append(rand_intersections)

      init_population = [Chromosome(content, self.fitness(content,3)) for content in init_population]
      self.top_chromosome = min(init_population, key=lambda chromo: chromo.fitness)
      print("Current top solution: %s" % self.top_chromosome)
      return init_population

    def selection(self, chromosomes):
      # print (self.i, '-- selection ',   self.fitness([5, 1, 14, 20, 19],3)   )
      self.i = self.i + 1

      """Ranking-based selection method"""

      # Chromosomes are sorted ascending by their fitness value  
      chromosomes.sort(key=lambda x: x.fitness)
      L = self.reproduction_size
      # selected_chromosomes = []
      selected_chromosomes = chromosomes[:L]
        
      # for i in range(self.reproduction_size):
      #   j = L - np.floor((-1 + np.sqrt(1 + 4*random.uniform(0, 1)*(L**2 + L))) / 2)
      #   selected_chromosomes.append(chromosomes[int(j)])
      return selected_chromosomes

    def create_generation(self, for_reproduction):
      
      # print (self.i , '-- create generation pop',   self.fitness([5, 1, 14, 20, 19],3)   )
      self.i = self.i + 1

      """
        Creates new generation from individuals that are chosen for reproduction, 
        by applying crossover and mutation operators. 
        Size of the new generation is same as the size of previous. 
        """
      new_generation = []
       
      while len(new_generation) < self.generation_size:
        parents = random.sample(for_reproduction, 2)
        child1, child2 = self.crossover(parents[0].content, parents[1].content)

        child1 = self.mutation(child1)
        new_generation.append(Chromosome(child1, self.fitness(child1, 3)))
            
        if child2 != None and len(new_generation) < self.generation_size:
          self.mutation(child2)
          new_generation.append(Chromosome(child2, self.fitness(child2, 3)))
            
      return new_generation

    def optimize(self):
      # print (self.i, '-- optimize pop',   self.fitness([5, 1, 14, 20, 19],3)   )
      self.i = self.i + 1

      start_time = time.time()
      chromosomes = self.initial_random_population()
      # print( optim1' ,  self.fitness([18, 16, 14, 15, 7], self.a ), chromosomes)
      while self.current_iteration < self.iterations:
        
        print("Iteration: %d" % self.current_iteration, )
        #From current population choose individuals for reproduction
        for_reproduction = self.selection(chromosomes)
        
        # Create new generation from individuals that are chosen for reproduction 
        chromosomes = self.create_generation(for_reproduction)
        print(chromosomes)
        self.current_iteration += 1
            
        chromosome_with_min_fitness = min(chromosomes, key=lambda chromo: chromo.fitness)
        if chromosome_with_min_fitness.fitness < self.top_chromosome.fitness:
          self.top_chromosome = chromosome_with_min_fitness
        print(f"Current top solution: {self.top_chromosome}! tot :  {self.tot}?")
        
            
      end_time = time.time()
      self.time = end_time - start_time
      hours, rem = divmod(end_time - start_time, 3600)
      minutes, seconds = divmod(rem, 60)
            
      print()
      print("Final top solution: %s" % self.top_chromosome)
      print('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

#%%
def evaluate_length_portion(a):
    if a == portion.empty():
        return 0
    if a.atomic:
        return a.upper - a.lower
    else:
        l = 0
        for i in range(len(a)):
            t = a[i]
            l = l + (  t.upper - t.lower )
        return l

#%%
window.mainloop()

#%%
# def multi_ga(deb,fin, pourcentage) :
#     results= []
#     num_rsu =  deb
    
#     gen_size = 3
#     reprod_size = 2
#     num_iter = 5
#     print(num_rsu, gen_size, reprod_size,num_iter)
#     test2 = GeneticAlgos(tot_sum_edges, aa, nodes, edges2, num_rsu ,  gen_size,      reprod_size,           num_iter ,        False, False)
#     test2.optimize()
    
#     while True:
#         num_rsu = num_rsu + 1

#         print(num_rsu, gen_size, reprod_size,num_iter)
#         test2 = GeneticAlgos(tot_sum_edges, aa, nodes, edges2, num_rsu ,  gen_size,      reprod_size,           num_iter ,        False, False)
#         test2.optimize() 
        
#         results.append(test2.top_chromosome)
#         pourcent = 100 - ( test2.top_chromosome.fitness *100 / test2.tot )
#         print('size :',size, 'num rsu :',num_rsu,'tot :', test2.tot, 'pourcent :',pourcent)
#         print('################# NEXT STEP :',num_rsu,'RSU ###################')
#         if pourcent > pourcentage :
#             break

#     print (results)
#     print ('Couverture Total :', test2.tot)
#     return results
# #%% open file
# folder_selected = 'test'

# #%% Start bouton

# # global rayon
# rayon =  500
# op_open_files()
# op_create_distance_matrix() 
# op_add_long_edges()
# op_adj_matrix()
# op_edg_node_matrix()
# op_coverage_of_all()


# #%%

# multi_ga(3,40,70)
# #%%Button Run GA

# num_rsu =  5
# gen_size = 3
# reprod_size = 2
# num_iter = 1
# print(num_rsu, gen_size, reprod_size,num_iter)
# test2 = GeneticAlgos(tot_sum_edges, aa, nodes, edges2, num_rsu ,  gen_size,      reprod_size,           num_iter ,        False, False)
# # test2.optimize()

# #%% plot GA

# print(test2.top_chromosome.content)
# plot_map(test2.top_chromosome.content,rayon)

