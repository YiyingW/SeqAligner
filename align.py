# Yiying Wang 10-1-2015

# alignment code works for both GLOBAL and LOCAL


# read input
import sys
import numpy as np
########################### READ INPUT FILE #########################
f = open(sys.argv[1], 'r')
data = []
for line in f:
	data.append(line.strip('\n'))

A = data[0]
B = data[1]
GorL = int(data[2])
da, ea, db, eb = map(float,data[3].split())
NA = int(data[4])
alphabetA = data[5]
NB = int(data[6])	
alphabetB = data[7]
dictA = {}
for i in range(0, NA):
	dictA[alphabetA[i]] = i
dictB = {}
for i in range(0, NB):
	dictB[alphabetB[i]] = i
# set up the match matrix
MatchMatrix = np.zeros([NA, NB])
for number in range(8, len(data)):
	if len(data[number]) > 0:
		entry = data[number].split()
		i = int(entry[0])
		j = int(entry[1])
		value = float(entry[4])
		MatchMatrix[i-1][j-1] = value
########################### END READ INPUT FILE#########################

nrow = len(A) + 1
ncol = len(B) + 1
M = np.zeros([nrow, ncol])
Ix = np.zeros([nrow, ncol])
Iy = np.zeros([nrow, ncol])
graph = {}

############ Define function used to find all paths #####################
class Queue:
	def __init__(self):
		self.items = []
	def is_empty(self):
		return self.items == []
	def enqueue(self, item):
		self.items.insert(0, item)
	def dequeue(self):
		return self.items.pop()
	def size(self):
		return len(self.items)
def BFS(graph, start, q):
	temp_path = [start]
	q.enqueue(temp_path)
	while q.is_empty() == False:
		temp_path = q.dequeue()
		last_node = temp_path[len(temp_path) - 1]
		if len(graph[last_node]) == 0:
			paths.append(temp_path)
		for link_node in graph[last_node]:
			if link_node not in temp_path:
				new_path = []
				new_path = temp_path + [link_node]
				q.enqueue(new_path)

def BFS_local(graph, start, q):
	temp_path = [start]
	q.enqueue(temp_path)
	while q.is_empty() == False:
		temp_path = q.dequeue()
		last_node = temp_path[len(temp_path) - 1]
		if len(graph[last_node]) == 0 and last_node[0] == 'M':
			paths.append(temp_path)
		for link_node in graph[last_node]:
			if link_node not in temp_path:
				new_path = []
				new_path = temp_path + [link_node]
				q.enqueue(new_path)
##################END Function defination ##################################

####################################### GLOBAL ALIGNMENT #####################################
# set up the equation 2.16 to calculate M, Ix and Iy
# also keep the track of paths in a graph, each node is a tuple of tuple, eg. (M,(i,j))
# the path must start from the first col and row of M and also end in the first col and row of M
if GorL == 0:
	for col in range(0, ncol):
		graph[('M', (0, col))] = []
		graph[('Ix', (0, col))] = []
		graph[('Iy', (0, col))] = []
	for row in range(1, nrow):
		graph[('M',(row, 0))] = []
		graph[('Ix',(row, 0))] = []
		graph[('Iy',(row, 0))] = []


	for i in range(0, len(A)):
		Ai = A[i]
		for j in range(0, len(B)):
			Bj = B[j]
			row_index = i + 1
			col_index = j + 1
			s_Ai_Bj = MatchMatrix[dictA[Ai]][dictB[Bj]]
			M[row_index][col_index] = round(max(M[row_index-1][col_index-1], Ix[row_index-1][col_index-1], Iy[row_index-1][col_index-1]) + s_Ai_Bj,1)
			if M[row_index][col_index] == round(M[row_index-1][col_index-1] + s_Ai_Bj,1):
				graph[('M',(row_index, col_index))] = [('M', (row_index-1, col_index-1))]
			if M[row_index][col_index] == round(Ix[row_index-1][col_index-1] + s_Ai_Bj,1):
				if ('M',(row_index, col_index)) not in graph.keys():
					graph[('M',(row_index, col_index))] = [('Ix', (row_index-1, col_index-1))]
				else:
					graph[('M',(row_index, col_index))].append(('Ix', (row_index-1, col_index-1)))
			if M[row_index][col_index] == round(Iy[row_index-1][col_index-1] + s_Ai_Bj,1):
				if ('M',(row_index, col_index)) not in graph.keys():
					graph[('M',(row_index, col_index))] = [('Iy', (row_index-1, col_index-1))]
				else:
					graph[('M',(row_index, col_index))].append(('Iy', (row_index-1, col_index-1)))
			Ix[row_index][col_index] = round(max(M[row_index-1][col_index]- db, Ix[row_index-1][col_index] - eb),1)
			if Ix[row_index][col_index] == round(M[row_index-1][col_index]- db,1):
				graph[('Ix',(row_index, col_index))] = [('M', (row_index-1, col_index))]
			if Ix[row_index][col_index] == round(Ix[row_index-1][col_index]- eb,1):
				if ('Ix',(row_index, col_index)) not in graph.keys():
					graph[('Ix',(row_index, col_index))] = [('Ix', (row_index-1, col_index))]					
				else:
					graph[('Ix',(row_index, col_index))].append(('Ix', (row_index-1, col_index)))		
			Iy[row_index][col_index] = round(max(M[row_index][col_index-1] - da, Iy[row_index][col_index-1] - ea),1)
			if Iy[row_index][col_index] == round(M[row_index][col_index-1]- da,1):
				graph[('Iy',(row_index, col_index))] = [('M', (row_index, col_index-1))]
			if Iy[row_index][col_index] == round(Iy[row_index][col_index-1]- ea,1):
				if ('Iy',(row_index, col_index)) not in graph.keys():
					graph[('Iy',(row_index, col_index))] = [('Iy', (row_index, col_index-1))]					
				else:
					graph[('Iy',(row_index, col_index))].append(('Iy', (row_index, col_index-1)))
# Find out the max score and it's locations from the last column and last row of M matrix
	Max_score_positions = []
	row_max = M[-1].max()
	MT = M.transpose()
	col_max = MT[-1].max()
	max_score = max(row_max, col_max)
	for col in range(0, ncol):
		if M[-1][col] == max_score:
			Max_score_positions.append(('M',(nrow-1,col)))
	for row in range(0, nrow):
		if M[row][-1] == max_score:
			Max_score_positions.append(('M', (row, ncol-1)))

# Find the paths from best score to the ends
	path_queue = Queue()
	paths = []
	for best_score_position in Max_score_positions:
		BFS(graph, best_score_position, path_queue)

#################################### END GLOBAL #########################################################


################################### LOCAL ALIGNMENT #####################################################
# set up the equation 2.16 to calculate M, Ix and Iy
# also keep the track of paths in a graph, each node is a tuple of tuple, eg. (M,(i,j))
# All three score matrix do NOT contain elements < 0
# Match matrix must have negative scores (satisfied)
# best score used to start traceback can come from any position in M matrix 
if GorL == 1:
	for col in range(0, ncol):
		graph[('M', (0, col))] = []
		graph[('Ix', (0, col))] = []
		graph[('Iy', (0, col))] = []
	for row in range(1, nrow):
		graph[('M',(row, 0))] = []
		graph[('Ix',(row, 0))] = []
		graph[('Iy',(row, 0))] = []
	for i in range(0, len(A)):
		Ai = A[i]
		for j in range(0, len(B)):
			Bj = B[j]
			row_index = i + 1
			col_index = j + 1
			s_Ai_Bj = MatchMatrix[dictA[Ai]][dictB[Bj]]
			# find the max of three possible values and compare it to 0 
			max_of_choices = round(max(M[row_index-1][col_index-1], Ix[row_index-1][col_index-1], Iy[row_index-1][col_index-1]) + s_Ai_Bj,1)
			if max_of_choices <= 0:
				M[row_index][col_index] = 0
				graph[('M',(row_index, col_index))] = []
			else:
				M[row_index][col_index] = max_of_choices
				if M[row_index][col_index] == round(M[row_index-1][col_index-1] + s_Ai_Bj,1):
					graph[('M',(row_index, col_index))] = [('M', (row_index-1, col_index-1))]
				if M[row_index][col_index] == round(Ix[row_index-1][col_index-1] + s_Ai_Bj,1):
					if ('M',(row_index, col_index)) not in graph.keys():
						graph[('M',(row_index, col_index))] = [('Ix', (row_index-1, col_index-1))]
					else:
						graph[('M',(row_index, col_index))].append(('Ix', (row_index-1, col_index-1)))
				if M[row_index][col_index] == round(Iy[row_index-1][col_index-1] + s_Ai_Bj,1):
					if ('M',(row_index, col_index)) not in graph.keys():
						graph[('M',(row_index, col_index))] = [('Iy', (row_index-1, col_index-1))]
					else:
						graph[('M',(row_index, col_index))].append(('Iy', (row_index-1, col_index-1)))


			max_of_choices = round(max(M[row_index-1][col_index]- db, Ix[row_index-1][col_index] - eb),1)
			if max_of_choices <= 0:
				Ix[row_index][col_index] = 0
				graph[('Ix',(row_index, col_index))] = []
			else:
				Ix[row_index][col_index] = max_of_choices			
				if Ix[row_index][col_index] == round(M[row_index-1][col_index]- db,1):
					graph[('Ix',(row_index, col_index))] = [('M', (row_index-1, col_index))]
				if Ix[row_index][col_index] == round(Ix[row_index-1][col_index]- eb,1):
					if ('Ix',(row_index, col_index)) not in graph.keys():
						graph[('Ix',(row_index, col_index))] = [('Ix', (row_index-1, col_index))]					
					else:
						graph[('Ix',(row_index, col_index))].append(('Ix', (row_index-1, col_index)))		
			
			max_of_choices = round(max(M[row_index][col_index-1] - da, Iy[row_index][col_index-1] - ea),1)
			if max_of_choices <= 0:
				Iy[row_index][col_index] = 0
				graph[('Iy',(row_index, col_index))] = []
			else:
				Iy[row_index][col_index] = max_of_choices
				if Iy[row_index][col_index] == round(M[row_index][col_index-1]- da,1):
					graph[('Iy',(row_index, col_index))] = [('M', (row_index, col_index-1))]
				if Iy[row_index][col_index] == round(Iy[row_index][col_index-1]- ea,1):
					if ('Iy',(row_index, col_index)) not in graph.keys():
						graph[('Iy',(row_index, col_index))] = [('Iy', (row_index, col_index-1))]					
					else:
						graph[('Iy',(row_index, col_index))].append(('Iy', (row_index, col_index-1)))

# Find out the max score and it's locations from the whole M matrix
	Max_score_positions = []
	max_score = M.max()
	for col in range(0, ncol):
		for row in range(0, nrow):
			if M[row][col] == max_score:
				Max_score_positions.append(('M', (row, col)))


# Find the paths from best score to the ends
	path_queue = Queue()
	paths = []
	for best_score_position in Max_score_positions:
		BFS_local(graph, best_score_position, path_queue)

#################################### END LOCAL #########################################################



# paths found before need to be filtered to remove the ones end in Ix or Iy
filtered_paths = []
for path in paths:
	if path[-1][0] == 'Ix' or path[-1][0] == 'Iy':
		pass
	else:
		filtered_paths.append(path)
# given the paths, output the alignments 
def path_to_strings(path, A, B): # transform the path into result sequences
	# one path corresponding to one pair of alignment  
	new_A = ''
	new_B = ''
	start = path[0]
	i = start[1][0]
	j = start[1][1]
	gap_indicator = start[0]
	for next_node in range(1, len(path)):
		next = path[next_node]
		if gap_indicator == 'M':
			new_A = A[i-1] + new_A
			new_B = B[j-1] + new_B
			i = i -1
			j = j - 1
		elif gap_indicator == 'Iy':
			new_A = '_' + new_A
			new_B = B[j-1] + new_B
			j = j - 1
		elif gap_indicator == 'Ix':			
			new_A = A[i-1] + new_A
			new_B = '_' + new_B
			i = i - 1
		gap_indicator = next[0]
	return new_A, new_B
alignments = set()
for path in filtered_paths:
	alignments.add(path_to_strings(path, A, B))


###### output file #######

output = open(sys.argv[2], 'w+')

output.write(str(max_score))

for pair in alignments:
	output.write('\n')
	output.write('\n')
	output.write(pair[0])
	output.write('\n')
	output.write(pair[1])
output.write('\n')
output.close()









