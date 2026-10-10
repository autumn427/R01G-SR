"""Exact-title-only structure repair; never infer facts or evidence IDs."""
from copy import deepcopy
from .data import SECTION_TITLES, SUBSECTION_VOCAB


def repair(report):
    owners={}
    for sid,titles in SUBSECTION_VOCAB.items():
        for title in titles:
            owners.setdefault(title,[]).append(sid)
    result=deepcopy(report)
    sections=result.get('sections')
    if not isinstance(sections,list):
        raise ValueError('sections must be a list')
    by_id={}
    for section in sections:
        sid=str(section.get('id',''))
        if sid not in SECTION_TITLES or sid in by_id:
            raise ValueError('invalid or duplicate section ID')
        by_id[sid]=section
    moves=[]
    pending=[]
    affected=set()
    for section in sections:
        sid=str(section['id'])
        kept=[]
        for sub in section.get('subsections',[]):
            destinations=owners.get(sub.get('title'),[])
            if len(destinations)==1 and destinations[0]!=sid:
                dest=destinations[0]
                pending.append((dest,sub))
                moves.append({'from':sid,'to':dest,'title':sub['title'],'bullets':len(sub.get('bullets',[]))})
                affected.update((sid,dest))
            else:
                kept.append(sub)
        section['subsections']=kept
    if not moves:
        return deepcopy(report),[]
    for sid,sub in pending:
        if sid not in by_id:
            by_id[sid]={'id':sid,'title':SECTION_TITLES[sid],'subsections':[]}
        existing=next((x for x in by_id[sid]['subsections'] if x['title']==sub['title']),None)
        if existing is None:
            by_id[sid]['subsections'].append(sub)
        else:
            existing.setdefault('bullets',[]).extend(sub.get('bullets',[]))
    for sid in affected:
        section=by_id[sid]
        section['title']=SECTION_TITLES[sid]
        for i,sub in enumerate(section['subsections']):
            if i>=26:
                raise ValueError('too many subsections')
            sub['id']=sid+chr(97+i)
            for j,bullet in enumerate(sub.get('bullets',[]),1):
                bullet['id']=sub['id']+'.'+str(j)
    result['sections']=[by_id[sid] for sid in sorted(by_id,key=int) if by_id[sid].get('subsections')]
    return result,moves
