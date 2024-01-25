#First example 
dta <- read.csv("/Users/komorebi/SSNA Fall 2023/11.03/User_to_User_Friendships.csv")
fix(dta)
rownames(dta)<-dta[,1]
dta2<-dta[,-1] # this drops the first column of data, which is duplicates of the first row

dim(dta2)

library(igraph)
#testing whether or not we has a symmetric matrix
dta1 <- mat_g2_incidence_to_1
dta2 <- mat_g2_incidence_to_1
table(rowSums(dta2)==colSums(dta2))

dta2<-as.matrix(dta2)

G<-graph.adjacency(dta2, mode=c("undirected"))#otherwise double counting
plot(G)

#Data frame creation
cent<-data.frame(bet=betweenness(G, normalized=F)/max(betweenness(G, normalized=F)),eig=evcent(G)$vector) # evcent returns lots of data associated with the EC, but we only need the leading eigenvector
rownames(cent)<-rownames(dta) #Ids in this case
cent

# cent<-read.csv("cent.csv")
# rownames(cent)<-rownames(dta)

residuals<-lm(eig~bet,data=cent)$residuals # We just save the residuals from OLS which will be used in the next steps
cent<-transform(cent,res=residuals) # This modifies the database called cent by adding a row with the residuals

# We have all we need to plot the data
# install.packages("ggplot2")
library(ggplot2) #Powerful but a little bit complicated platform

# Spelling out, cent is the dataset and we specifically are asking for two variables eig and bet, in addition we need the row names to be considered in the plot. Additionally I am asking the nodes or names to be weighed by the size of the residuals and colored by that as well.
# We use the residuals to color and shape the points of our plot, making it easier to spot outliers.
# x11()
pdf("key actor.pdf", 15, 15)
p<-ggplot(cent,aes(x=bet,y=eig, label=rownames(cent),colour=eig, size=(1/abs(res))))+xlab("Betweenness Centrality")+ylab("Eigenvector Centrality")
# p is the graph all the information necessary to, yet it requires some extra commands called adding layers
# geom_point() will add bubbles
p+geom_point()+labs(title="Key Actor Analysis for WTCC") 
# geom_text() adds names instead
p+geom_text()+labs(title="Key Actor Analysis for WTCC")
# We, of course cn add both, just adjusting them so that they do not overlap
p + geom_point() + geom_text(hjust=1.5, vjust=1)+labs(title="Key Actor Analysis for WTCC") 
#We can add the best line possible based on the data, to do this we need to compute the linear regression one more time and save the $\alpha$ and $\beta$ coefficients
coeffs<-as.data.frame(coef(lm(eig~bet,data=cent)))
#Then we just simply add the new variables to our old graph
p + geom_point() + geom_text(hjust=1.5, vjust=1)+labs(title="Key Actor Analysis for WTCC") + geom_abline(intercept = coeffs[1,], slope = coeffs[2,],colour = "red", size = 2,alpha=.25) 
#Finally, we can get rid of the legend since the regression line is doing a good job guiding us to detect outliers
p + geom_point() + geom_text(hjust=1.5, vjust=1)+labs(title="Key Actor Analysis for WTCC") + geom_abline(intercept = coeffs[1,], slope = coeffs[2,],colour = "red", size = 2,alpha=.25) + theme(legend.position = "none")
dev.off()

example <- p + geom_point() + geom_text(hjust=1.5, vjust=1)+labs(title="Key Actor Analysis for WTCC") + geom_abline(intercept = coeffs[1,], slope = coeffs[2,],colour = "red", size = 2,alpha=.25) + theme(legend.position = "none")
install.packages("plotly")
library(plotly)
ggplotly(example)

#####################################################################
	###################Figure 2##############################

# library(igraph)
# G<-graph.adjacency(dta2, mode=c("undirected"))
# cent<-data.frame(bet=betweenness(G),eig=evcent(G)$vector)
# rownames(cent)<-rownames(dta) #Ids in this case
# residuals<-lm(eig~bet,data=cent)$residuals
# cent<-transform(cent,res=residuals)
set.seed(47)
l<-layout.fruchterman.reingold(G, niter=5000)
V(G)$name<-rownames(dta2)
V(G)$size<-abs((cent$bet)/max(cent$bet))*15 #The divisor is the highest betweenness
nodes<-V(G)$name # Setting a variable to manipulate names, nodes contains the IDs of the participants

x<-quantile(cent$eig, .97)
nodes[which(abs(cent$eig)<(x))]<-NA # this gives the top 10%

#How many nodes would have names?
table(is.na(nodes))

plot(G,layout=l,vertex.label=nodes, vertex.label.dist=0.25, vertex.label.color="red",edge.width=1)
pdf("actor_plot.pdf") 
plot(G,layout=l,vertex.label=nodes, vertex.label.dist=0.25,
vertex.label.color="red",edge.width=1)
title(main="Key Actor Analysis", sub="Key actors weighted by BC, names are top 3% EC", col.main="black", col.sub="black", cex.sub=1.2,cex.main=2,font.sub=2)
dev.off()

###Second example###

# Isolates?
a <- rowSums(dta2)
table(a)
# Removing isolates manually
#This will render the matrix to be analyzed
dta2<-dta2[rowSums(dta2)!=0,]
dta2<-dta2[,colSums(dta2)!=0]
dim(dta2)

dta2 <- as.matrix(dta2)

G<-graph.adjacency(dta2, mode=c("undirected"))
cent<-data.frame(bet=betweenness(G, normalized=T),eig=evcent(G)$vector) # evcent returns  lots of data associated with the EC, but we only need the leading eigenvector
rownames(cent)<-rownames(dta2) #Ids in this case
fix(cent)

res<-lm(eig~bet,data=cent)$residuals # We just save the residuals from OLS which will be used in the next steps
cent<-transform(cent,res=res) # I modified the database called cent by adding a column with the residuals
# cent<-cent[cent$eig>0,]
# cent$eig<-abs(log(cent$eig))
# cent$bet<-abs(log(cent$bet))

# We have all we need to plot the data
install.packages("ggplot2")
library(ggplot2) #Powerful but complicated platform
# We use ggplot2 to make things prettier
# Spelling out, cet is the dataset and we specifically are asking for two variables eig and bet, in addition we need the row names to be considered in the plot. Additionally I am asking the bubbles or names to be weighed by the size of the residuals and colored by that as well.
# We use the residuals to color and shape the points of our plot, making it easier to spot outliers.
p<-ggplot(cent,aes(x=bet,y=eig, label=rownames(cent),colour=res, size=abs(res)))+xlab("Betweenness Centrality")+ylab("Eigenvector Centrality")
# p is the graph all the information necessary to, yet it requires some extra commands called adding layers
# geom_point() will add bubbles
p+geom_point()+labs(title="Key Actor Analysis for WTCC") 
# geom_text() adds names instead
p+geom_text()+labs(title="Key Actor Analysis for Tri-C")
# We, of course cn add both, just adjusting them so that they do not overlap
p + geom_point() + geom_text(hjust=2, vjust=2)+labs(title="Key Actor Analysis for Tri-C") 
#We can add the best line possible based on the data, to do this we need to compute the linear regression one more time and save the $\alpha$ and $\beta$ coefficients
coeffs<-as.data.frame(coef(lm(eig~bet,data=cent)))
#Then we just simply add the new variables to our old graph
p + geom_point() + geom_text(hjust=2, vjust=2)+labs(title="Key Actor Analysis for WTCC") + geom_abline(intercept = coeffs[1,], slope = coeffs[2,],colour = "red", size = 2,alpha=.25) 
#Finally, we can get rid of the legend since the regression line is doing a good job guiding us to detect outliers
p + geom_point() + geom_text(hjust=1.5, vjust=1)+labs(title="Key Actor Analysis for Tri-C") + geom_abline(intercept = coeffs[1,], slope = coeffs[2,],colour = "red", size = 2,alpha=.25) + theme(legend.position = "none")

#####################################################################
	###################Figure 2##############################

library(igraph)
l <- layout_with_kk(G)
V(G)$name<-rownames(dta2)
V(G)$size<-abs((cent$bet)/max(cent$bet))*15 #The divisor is the highest betweenness
nodes<-V(G)$name # Setting a variable to manipulte names, nodes contains the IDs of the participants

x<-quantile(cent$eig, .975)
nodes[which(abs(cent$eig)<=(x))]<-NA # this gives the top 10%

V(G)$color <- rgb(0, 139, 0, max=255, 255/2)
V(G)$frame.color <- NA
E(G)$color <- rgb(161, 161, 161, max=255, 255/2)
#How many nodes would have names?
table(is.na(nodes))
plot(G,layout=l,vertex.label=nodes, vertex.label.dist=0.25,vertex.label.cex=0.25, vertex.label.color=rgb(0, 104, 139, max=255, 255/3),edge.width=1)

pdf("actor_plot no isolates.pdf", 25, 25) 
plot(G,layout=l,vertex.label=nodes, vertex.label.dist=0.0,vertex.label.cex=0.25, vertex.label.color=rgb(0, 104, 139, max=255, 255/3),edge.width=1)
title(main="Key Actor Analysis LATTC", sub="Key actors weigthed by eigenvector and betweenness centrality", col.main="black", col.sub="black", cex.sub=1.2,cex.main=2,font.sub=2)
dev.off()

