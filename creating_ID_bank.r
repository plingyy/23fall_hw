#clean data, prepare author names that match ID, get one-mode matrix g and two-mode matrix g2 
library(splitstackshape)
library(igraph)

source <- read.csv("scopus.csv")

# clean the data, keep only authors and articles
source <- source[source$Author.s..ID!="[No author id available]",]

#Addressing names and titles
#Create labels for each author into V(g)
source$au <- gsub(" Jr.,", "", 
                  gsub(" II.,", "", 
                       gsub(" Jr.", "",
                            gsub(" M.S.", "",
                                 gsub(" M.S.,", "",
                                      gsub(" II.", "", source[,1]))))))

source$au <- gsub("\\.,", ";", source$au)
source$au <- tolower(source$au)


# Creating adjacency list from names
df_all_authors <- as.data.frame(source[ , ncol(source)]) #use ncol(data.frame) to get the last column of the data.frame
colnames(df_all_authors) <- "AU"


#As can be seen in the file the separator of interest is authors' name:
authornames_split <- cSplit(df_all_authors, splitCols = "AU", sep = ";", direction = "wide", drop = TRUE) #retain the matrix form version of the adjacency list input

dim(authornames_split)

#Adjacency list from authors' IDs, that is, split the authorsID column
df_authors_id <- as.data.frame(source[ , 3])

colnames(df_authors_id) <- "AU"

author_id_split <- cSplit(df_authors_id, splitCols = "AU", sep = ";", direction = "wide", drop = TRUE) 

dim(author_id_split)

#If dimensions of Id and name adjacency list match, we are good if not we need to find differences
df_authorid_authorname_unlisted <- data.frame(id = unlist(author_id_split), names = unlist(authornames_split))

df_authorid_authorname_unlisted <- df_authorid_authorname_unlisted[!is.na(df_authorid_authorname_unlisted$id),]

df_authorid_authorname_unlisted <- df_authorid_authorname_unlisted[!duplicated(df_authorid_authorname_unlisted$id),]

#most Prolific authors by ID
pub_count <- as.data.frame(table(unlist(author_id_split)))

df_authorid_authorname_unlisted$pub_count <- pub_count$Freq[match(df_authorid_authorname_unlisted$id, pub_count$Var1)] 

head(df_authorid_authorname_unlisted[order(df_authorid_authorname_unlisted$pub_count, decreasing=T),])

a <- df_authorid_authorname_unlisted
a <- a[order(a$pub_count, decreasing = T),]
head(a, 10)

###head(as.data.frame(table(unlist(author_id_split)))[order(as.data.frame(table(unlist(author_id_split)))[,2], decreasing=T),])

#In case you need to add names to your graphs
V(g2)$label <- df_authorid_authorname_unlisted$name[match(V(g2)$name, df_authorid_authorname_unlisted$id)]

