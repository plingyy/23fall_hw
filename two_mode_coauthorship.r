#clean data, prepare author names that match ID, get one-mode matrix g and two-mode matrix g2 
library(splitstackshape)
library(igraph)

source <- read.csv("/Users/komorebi/SSNA Fall 2023/10.06/scopus.csv")

# clean the data, keep only authors and articles
source <- source[source$Author.s..ID!="[No author id available]",]

### create a author-author matrix through transformation of a two-mode edgelist#
authors <- as.data.frame(source[,3])

colnames(authors) <- "AU"

source_split <- cSplit(authors, splitCols = "AU", sep = ";", direction = "wide", drop = TRUE)
#No needed but just in case
# source_split <- data.frame(lapply(source_split, trimws), stringsAsFactors = FALSE)
mat_source_split <- as.matrix(source_split)

combined <- cbind(source$EID, mat_source_split)

mat_combined_eid_source_split <- as.matrix(combined)

edgelist_two_mode <- cbind(mat_combined_eid_source_split[, 1], c(mat_combined_eid_source_split[, -1]))
edgelist_two_mode <- edgelist_two_mode[!is.na(edgelist_two_mode[,2]), ]

head(edgelist_two_mode)

g2 <- graph.edgelist(edgelist_two_mode[, 2:1], directed = TRUE)
g2
g2 <- graph.edgelist(edgelist_two_mode[, 2:1], directed = FALSE) 
g2
# which one is it?

V(g2)$type <- V(g2)$name %in% edgelist_two_mode[ , 2]
table(V(g2)$type)
i<-table(V(g2)$type)[2]

#In case you need to add names to your graphs
V(g2)$label <- a$names[match(V(g2)$name, a$id)]
table(is.na(V(g2)$label))

V(g2)$label <- ifelse(is.na(V(g2)$label), V(g2)$name, V(g2)$label)
table(is.na(V(g2)$label))

#Transformations to retain actors

mat_g2_incidence <- t(get.incidence(g2))
mat_g2_incidence_to_1 <- mat_g2_incidence%*%t(mat_g2_incidence)

diag(mat_g2_incidence_to_1)<-0

g <- graph.adjacency(mat_g2_incidence_to_1, mode = "undirected")
plot(g)

# Install and load the igraph library
install.packages("igraph")
library(igraph)

# Create a bipartite graph
nodes_set_A <- c(1, 2, 3)
nodes_set_B <- c('A', 'B', 'C')

edges <- data.frame(from = c(1, 1, 2, 3),
                    to = c('A', 'B', 'A', 'C'))

g <- graph_from_data_frame(edges, directed = FALSE)

# Plot the bipartite graph
plot(g, layout = layout.bipartite(g, types = bipartite_mapping(g)$type),
     vertex.color = c("skyblue", "lightgreen")[V(g)$type + 1],
     vertex.label.color = "black", 
     vertex.label.dist = 0.5,
     vertex.label.cex = 1.5,
     edge.color = "black",
     main = "Bipartite Graph")

# Add legend
legend("bottomright", legend = c("Set A", "Set B"), fill = c("skyblue", "lightgreen"))

# Display the plot
