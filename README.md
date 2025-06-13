run
 	g++ cluster.cpp clustering.cpp
	./a.out *_data.csv
	g++ draw_graph.cpp
	./a.out *_data.csv
	python3 combine_image_with_labels.py *_data.result
