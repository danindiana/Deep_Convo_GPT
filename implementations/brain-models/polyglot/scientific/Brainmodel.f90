program BrainModel
    implicit none
    integer, dimension(10) :: encodedData
    integer :: i, input

    do i=1,10
        print*, 'Enter sensory input data (integer from 1 to 10) for memory ', i, ': '
        read*, input
        encodedData(i) = input
    end do

    print*, 'Retrieval process:'
    do i=1,10
        print*, 'Retrieved data from memory ', i, ' is ', encodedData(i)
    end do

end program BrainModel
