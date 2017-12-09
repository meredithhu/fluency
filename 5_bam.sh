sqlite3 bam.sqlite <<EOF | parallel -C'\|' 'mkdir -p {2}; wget --wait=1 {1} -O {2}/{3}.jpg'
    select src, attribute, mid
    from modules, crowd_labels where modules.id = crowd_labels.mid
    and label="positive" 
    and (attribute = "media_comic" or attribute = "media_vectorart" or attribute = "media_3d_graphics" or attribute = "media_watercolor");
EOF