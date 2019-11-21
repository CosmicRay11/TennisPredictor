# score parser
# returns None for walkover/retired and the number of sets won by each player otherwise


def parse_score(score):

    score = str(score)

    if "RET" in score or "W/O" in score or "Walkover" in score or "DEF" in score:
        return None

    # attempt to get a meaningful score from the score string
    try:
        # split the score by sets
        sets = score.split(' ')
        wSets = 0
        for s in sets:
            # split the sets by games won by winner and loser
            w, l = s.split('-')
            
            # if set had a tie break, get rid of the tie-break part
            # e.g. 7-6(7-5) goes to 7-6
            if '(' in l:
                l = l.split('(')[0]

            # if set is bracketed, get rid of the square brackets (used for championship tie-breaks)
            if '[' in w:
                w = w[1:]
                l = l[:-1]

            # if the winner won more games than the loser, add one the number of sets won by the winner
            if int(w) > int(l):
                wSets += 1

        return wSets, len(sets) - wSets

    # if an error is thrown, the score must be faulty and so return None to represent an invalid score
    except:
        return None
