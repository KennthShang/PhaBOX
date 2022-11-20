library(networkD3)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

root <- args[1]
edge <- read.csv(args[2])
node <- read.csv(args[3])
num <- args[4]
tool <- args[5]


# Plot
forceNetwork(Links = edge, Nodes = node,
             Source = "Source", Target = "Target",
             Value = "Weight", NodeID = "Label", Nodesize= "Size",
             Group = "Category", opacity = 0.8, zoom = TRUE, legend=TRUE) %>%
saveNetwork(file = paste(root, '/html_', tool, '/Net_',num,'.html', sep=""))

#flag = try(forceNetwork(Links = edge, Nodes = node,
#             Source = "Source", Target = "Target",
#             Value = "Weight", NodeID = "Label", Nodesize= "Size",
#             Group = "Category", opacity = 0.8, zoom = TRUE, legend=TRUE) %>%
#             saveNetwork(file = paste(root, '/html_', tool, '/Net_',num,'.html', sep="")), silent=FALSE)
#if ('try-error' %in% class(flag)){
##    
#  }



