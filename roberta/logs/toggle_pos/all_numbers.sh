for i in $(ls | grep -v 'broken' | grep -v 'sh' | grep -v 'hi'); do
    lang=$(echo $i | awk -F'.' '{printf $1}')
    seed=$(echo $i | awk -F'.' '{printf $2}')
    printf "%s\t%s\t" $lang $seed
    for metric in 0_retrieve 8_retrieve 0_translate 8_translate ml_score full_perplexity; do
        cat $i | grep "$metric[^_]" | awk -F '\t' '{printf "%s\t", $2}'
    done 
    printf "\n"
done
