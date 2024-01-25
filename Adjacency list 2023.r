###########################################
###		Code for first deliverable      ###
###########################################

#Procedures to decompose an adjacency list into a weighted graph
a<-read.csv("/Users/komorebi/Downloads/scopus.csv")

#Keeping only column with author relationships
authors<-as.data.frame(a[,1])
colnames(authors)<-"AU"


#package for first split
# install.packages("splitstackshape")
library(splitstackshape)
#As can be seen in the file the separator of interest is :
a1<-cSplit(authors, splitCols = "AU", sep = ";", direction = "wide", drop = FALSE) #retain the matrix form version of the adjacency list input
#Here we just drop the original first column
fix(a1)
a1<-a1[,-1]
class(a1)

# In case package cannot be installed uncomment and keep working using this version, alternatively you can decompose the cells into columns using excel

#read it as a matrix
mat <- as.matrix(a1)
mat

mat<-tolower(mat)

dim(mat)# the resulting column dimension is the number of times you will have to repeat the following procedure minus 1

#all combinations except first author with itself
edgelist1 <- cbind(mat[, 1], c(mat[, -1]))
edgelist1

edgelist2 <- cbind(mat[, 2], c(mat[, -c(1:2)]))
edgelist2

edgelist3 <- cbind(mat[, 3], c(mat[, -c(1:3)]))
edgelist3

edgelist4 <- cbind(mat[, 4], c(mat[, -c(1:4)]))
edgelist4

edgelist5 <- cbind(mat[, 5], c(mat[, -c(1:5)]))
edgelist5

edgelist6 <- cbind(mat[, 6], c(mat[, -c(1:6)]))
edgelist6

edgelist7 <- cbind(mat[, 7], c(mat[, -c(1:7)]))
edgelist7

# edgelist8 <- cbind(mat[, 8], c(mat[, -c(1:8)]))
# edgelist8

# edgelist9 <- cbind(mat[, 9], c(mat[, -c(1:9)]))
# edgelist9

edgelist <- rbind(edgelist1, edgelist2, edgelist3) 
dim(edgelist)

#if reading a1 from csv
# First approach
# # edgelist$V2[edgelist$V2==""]<-NA
# # #second approach
# # edgelist<-edgelist[edgelist[,2]!="",]

edgelist<-edgelist[!is.na(edgelist[,2]),]
dim(edgelist)


a1<-mat
edgelist1<-matrix(NA, 1, 2)#empty matrix two columns
for (i in 1:(ncol(a1)-1)) {
  edgelist11 <- cbind(a1[, i], c(a1[, -c(1:i)]))
  edgelist1 <- rbind(edgelist1,edgelist11)
  edgelist1<-edgelist1[!is.na(edgelist1[,2]),]
  edgelist1<-edgelist1[edgelist1[,2]!="",]
  }
dim(edgelist1)

install.packages("igraph")
library(igraph)
plot(graph.edgelist(edgelist1))

g<- graph.edgelist(edgelist1, directed = FALSE)

E(g)$weight	<- 1#must step
g.c <- simplify(g)
E(g.c)$weight 

links<-as.data.frame(cbind(get.edgelist(g.c), E(g.c)$weight))

head(links,20)
dim(links)
links$V3<-as.numeric(links$V3)

links<- links[order(links$V3, decreasing=T),]

table(links[,1])

###########################################
###		Code for second deliverable     ###
###########################################

#Add publication as first column
a1 <- cbind(a$EID, a1)

mat <- as.matrix(a1)
fix(mat)
edgelist_two_mode <- cbind(mat[, 1], c(mat[, -1]))
edgelist_two_mode
edgelist_two_mode <- edgelist_two_mode[!is.na(edgelist_two_mode[,2]), ]

g2<- graph.edgelist(edgelist_two_mode[, 2:1], directed = FALSE) #TRUE or FALSE

V(g2)$type <- V(g2)$name %in% edgelist_two_mode[,1]
degree(g2)[V(g2)$type==F]
sort(degree(g2)[V(g2)$type==F], decreasing =T)

#How manu authors and how many publications?
table(V(g2)$type)

#Transformation 
links2<-as.data.frame(cbind(get.edgelist(g2)))

###########################################
###		Code for third deliverable     ###
###########################################
# Transformations
mat2 <- get.adjacency(g2) 

mat2_to_1<-mat2%*%t(mat2)

