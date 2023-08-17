function fitness = Fitness(x_pop,dataset,K_C)
[pop_num,~] = size(x_pop);

fitness = zeros(pop_num,1);

	for i = 1:pop_num
	    [fitness(i),~] = assignment(x_pop(i,:),dataset,K_C);
	end
end
