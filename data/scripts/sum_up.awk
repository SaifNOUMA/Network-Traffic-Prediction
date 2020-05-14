BEGIN{diff=$1}
{
	step=$1-diff
	quantity+=$2
	len=$2
       	
	#printf("%Step = %20.19f\t %20.19f\t %20.19f\n",step,diff, $1)
	#printf("Granularity %d\n", TimeStep)
	if(step>granularity)
	{
		diff=$1-diff
       		#printf("%20.19f\t %20.19f\n ", quantity, diff)
		printf("%d\n", quantity)
		quantity=0
		diff=$1
	}
}

