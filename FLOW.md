flowchart TD
    START([User message]) --> ROUTER{Router: classify intent}

    %% DISCOVER
    ROUTER -->|discover| FIND1[find_relevant_papers]
    FIND1 --> FETCH1[fetch_papers]
    FETCH1 --> DISPLAY[Display paper list]
    DISPLAY --> END1([End])

    %% SUMMARIZE
    ROUTER -->|summarize| FILECHECK{File attached?}
    FILECHECK -->|no| INTERRUPT[Interrupt: ask user to upload]
    INTERRUPT -.->|resumes on next turn| FILECHECK
    FILECHECK -->|yes| EXTRACT1[extract_formatted]
    EXTRACT1 --> SUMM[summarise_paper]
    SUMM --> CRITIC{summary_critic: pass?}
    CRITIC -->|fail, attempts < max| SUMM
    CRITIC -->|pass| END2([End])
    CRITIC -->|fail, attempts >= max| BESTEFFORT[Return best attempt + caveat]
    BESTEFFORT --> END2

    %% COMPARE
    ROUTER -->|compare| SOURCE1{Uploaded or search?}
    SOURCE1 -->|uploaded| EXTRACT2[extract_formatted per paper]
    SOURCE1 -->|search| FIND2[find_relevant_papers]
    FIND2 --> FETCH2[fetch_papers]
    FETCH2 --> EXTRACT2
    EXTRACT2 --> COMPAT{check_compatibility: 2-5 papers, comparable?}
    COMPAT -->|no| CLARIFY[Ask user to adjust selection]
    CLARIFY --> END3a([End])
    COMPAT -->|yes| COMPARE[compare_papers]
    COMPARE --> WANTDIGEST{User requested digest?}
    WANTDIGEST -->|yes| DIGEST[comparison_summariser]
    DIGEST --> END3([End])
    WANTDIGEST -->|no| END3

    %% VALIDATE
    ROUTER -->|validate| SOURCE2{Uploaded or search?}
    SOURCE2 -->|uploaded| EXTRACT3[extract_formatted]
    SOURCE2 -->|search| FIND3[find_relevant_papers]
    FIND3 --> FETCH3[fetch_papers]
    FETCH3 --> EXTRACT3
    EXTRACT3 --> INDEX[Chunk + embed into vector store]
    INDEX --> VALIDATE[validate_and_query]
    VALIDATE --> END4([End])