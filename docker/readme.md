# generate the docker to create tfrecords

## to run :  
`docker build -t imagename .`

## details : 
 in this process I'm using almost the same Dockerfile from [google TPU repo  (github.com)](https://github.com/tensorflow/tpu/tree/master/tools/docker) 

the main process needs data to be in the following format : 

train / 
	&nbsp;&nbsp; - class1 
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- image1
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- image2 
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- image3
	&nbsp;&nbsp;- class2
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- image1
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- image2 
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- image3
test / 
etc ..

but to use csv files I made the main file that convert the csv to this format and then run the orginal code..
your csv file needs to be 
SET , IMAGE , CLASS 