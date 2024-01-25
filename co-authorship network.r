###########################################
###		Code for seventh deliverable      ###
###########################################
#Procedures to decompose an adjacency list into a weighted graph
a<-read.csv("scopus.csv")

library(stringr)
x <- "Jangjarat K.; Kraiwanit T.; Limna P.; Sonsuphap R."
str_remove_all(x, "[^[\\da-zA-Z ]]")

#Keeping only column with author relationships
authors<-as.data.frame(a[,1])#"Author.s..ID"])
colnames(authors)<-"AU"

#package for first split
# install.packages("splitstackshape")
library(splitstackshape)
authors$AU <- str_remove_all(authors$AU, "[^[\\a-zA-Z ]]")

#As can be seen in the file the separator of interest is :
a1<-cSplit(authors, splitCols = "AU", sep = ";", direction = "wide", drop = FALSE) #retain the matrix form version of the adjacency list input
#Here we just drop the original first column
a1<-a1[,-1]
class(a1)

#read it as a matrix
mat <- as.matrix(a1)
# mat

dim(mat)# the resulting column dimension is the number of times you will have to repeat the following procedure minus 1

a1<-mat
edgelist1<-matrix(NA, 1, 2)#empty matrix two columns
for (i in 1:(ncol(a1)-1)) {
  edgelist11 <- cbind(a1[, i], c(a1[, -c(1:i)]))
  edgelist1 <- rbind(edgelist1,edgelist11)
  edgelist1<-edgelist1[!is.na(edgelist1[,2]),]
  edgelist1<-edgelist1[edgelist1[,2]!="",]
  }
dim(edgelist1)

# install.packages("igraph")
library(igraph)

g<- graph.data.frame(edgelist1, directed = FALSE)

E(g)$weight	<- 1 #must step
g.c <- simplify(g)
E(g.c)$weight 

#centrality measures
cent<-data.frame(ID=V(g.c)$name, ev=evcent(g.c)$vector, deg=degree(g.c)/max(degree(g.c)), bet=betweenness(g.c, normalized=F)/max(betweenness(g.c, normalized=F)), clo=closeness(g.c)/max(closeness(g.c))) 
head(cent)
cent$max_cent <- cent$ev + cent$deg + cent$bet + cent$clo

cent<- cent[order(cent$max_cent, decreasing=T),] 
head(cent)

library(networkD3)
V(g.c)$label <- V(g.c)$name
V(g.c)$name<-1:length(V(g.c)) #(1:length(V(g)))-1
links<-as.data.frame(cbind(get.edgelist(g.c),as.numeric(E(g.c)$weight)))

links$V1<-as.numeric(as.character(links$V1))
links

links$V2<-as.numeric(as.character(links$V2))
str(links)

links
links<-cbind(links[,1:2]-1, links[,3])
colnames(links)<-c("source","target", "value")

#get publication count
pub_count <- as.data.frame(table(unlist(a1)))

#Adding attributes at the actor level
V(g.c)$max_cent <- cent$max_cent[match(V(g.c)$label, cent$ID)]
V(g.c)$pub_count <- pub_count$Freq[match(V(g.c)$label, pub_count$Var1)]

nodes <- data.frame(name=c(paste("ID: ", 
                                 V(g.c)$label,
                                 ", Pub count: ", 
                                 V(g.c)$pub_count,
                                 ", Max cent: ", 
                                 V(g.c)$max_cent, sep="")),
                    group = edge.betweenness.community(g.c)$membership, 
                    size=betweenness(g.c,directed=F,normalized=T)/max(betweenness(g.c,directed=F,normalized=T))*115) #so size isn't tiny

netviz <- forceNetwork(Links = links, Nodes = nodes,
                  Source = 'source', Target = 'target',
                  NodeID = 'name',
                  Group = 'group', # color nodes by group calculated earlier
                  charge = -5, # node repulsion
                  linkDistance = 20,
                  opacity = 1,
				  Value = 'value',
				  Nodesize = 'size', # color nodes by group calculated earlier
                  zoom = T, 
                  fontSize=24,
				  colourScale = JS("d3.scaleOrdinal(d3.schemeCategory20)"))
library(magrittr)
library(htmlwidgets)
library(htmltools)

HTMLaddons <- 
"function(el, x) { 
d3.select('body').style('background-color', ' #C0C0C0')
d3.selectAll('.legend text').style('fill', 'white') 
 d3.selectAll('.link').append('svg:title')
      .text(function(d) { return 'Intensity: ' + d.value ; })
  var options = x.options;
  var svg = d3.select(el).select('svg')
  var node = svg.selectAll('.node');
  var link = svg.selectAll('link');
  var mouseout = d3.selectAll('.node').on('mouseout');
  function nodeSize(d) {
    if (options.nodesize) {
      return eval(options.radiusCalculation);
    } else {
      return 6;
    }
  }

  
d3.selectAll('.node').on('click', onclick)

  function onclick(d) {
    if (d3.select(this).on('mouseout') == mouseout) {
      d3.select(this).on('mouseout', mouseout_clicked);
    } else {
      d3.select(this).on('mouseout', mouseout);
    }
  }

  function mouseout_clicked(d) {
    node.style('opacity', +options.opacity);
    link.style('opacity', +options.opacity);

    d3.select(this).select('circle').transition()
      .duration(750)
      .attr('r', function(d){return nodeSize(d);});
    d3.select(this).select('text').transition()
	
      .duration(1250)
      .attr('x', 0)
      .style('font', options.fontSize + 'px ');
  }

}
" 
netviz$x$links$linkDistance <- (1/links$value)*125
onRender(netviz, HTMLaddons) 

# ChatGPT co-authorship network
# Since December 2022, there has been 897 academic articles published on the topic ChatGPT. These articles have been co-authored by 2,118 academics, representing 8,173 collaborations.
# This network highlights the most influential authors as a function of connecting different articles together (i.e., betweenness centrality). Access to the original data available at https://cutt.ly/rwaClF9g. 
# ChatGPT_co-authorship_network 