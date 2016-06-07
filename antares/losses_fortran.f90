subroutine compute_bin_orderings(sorter, bin_indices, n_bins, sorter_bins, n_threads, local_bincounts)
    use omp_lib
    integer*4, intent(in) ::  sorter(:), bin_indices(:), sorter_bins, n_threads
    integer*4, intent(out) :: local_bincounts(size(sorter, 1))
    integer*4 :: subbincounter(0:sorter_bins, 0:n_bins)
    integer*4 :: j, order_grouping_size, sorter_bin, event_group
    integer*4 :: lower_bound, upper_bound
    subbincounter(:, :) = 0
    order_grouping_size = size(sorter, 1) / sorter_bins + 1
            
    call omp_set_num_threads(n_threads)        
            
    !$OMP PARALLEL DO SCHEDULE(STATIC) private(event_group)
    do j = lbound(sorter, 1), ubound(sorter, 1)
        event_group = sorter(j) / order_grouping_size
        subbincounter(event_group, bin_indices(j)) = &
            subbincounter(event_group, bin_indices(j)) + 1
    end do 
    
    do event_group = 2, sorter_bins
        subbincounter(event_group, :) = subbincounter(event_group, :) + subbincounter(event_group - 1, :)
    end do
            
    !$OMP PARALLEL DO SCHEDULE(STATIC) private(event_group)
    do j = lbound(sorter, 1), ubound(sorter, 1)
        event_group = sorter(j) / order_grouping_size
        local_bincounts(j) = subbincounter(event_group, bin_indices(j))
    end do 
    
end subroutine 