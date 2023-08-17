function [total_dist,res] = assignment(one_pop,dataset,K_C)
[data_num,dim] = size(dataset);

for k = 1:K_C
    res(k).mean(1,1:dim) = one_pop(1,dim*(k-1)+1:dim*k);
end


total_dist = 0; 
count = zeros(K_C,1);

for i = 1:data_num 
    temp_dist = inf;
    D = zeros(K_C,1);
    u = zeros(K_C,1);
    for k =1:K_C
        D(k) =abs(1-(dataset(i,:)*res(k).mean(1,1:dim)')^2) ;     
                                                              
        if(temp_dist > D(k))                                  
            count_flag = k;                                        
            temp_dist = D(k);
        end
    end

    
    for k =1:K_C
    u(k) =D(k)^-2*(sum(D.^-2)) ^-1;   
    end

    dist_record=D.^2.*u.^2;
        

    count(count_flag,1) = count(count_flag,1) + 1;    
    total_dist = total_dist + sum(dist_record);    
    res(count_flag).index(count(count_flag,1)) = i;           
    res(count_flag).dist(count(count_flag,1)) = temp_dist; 
end

end
