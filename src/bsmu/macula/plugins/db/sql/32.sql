UPDATE eyes
SET refraction = CASE refraction
    WHEN 'Em' THEN '1'
    WHEN 'Hm  cл' THEN '2'
    WHEN 'Hm  cр' THEN '3'
    WHEN 'Hm cл' THEN '2'
    WHEN 'Hm в' THEN 'Hm в'
    WHEN 'Hm сл' THEN '2'
    WHEN 'Hm сл.' THEN '2'
    WHEN 'Hm ср' THEN '3'
    WHEN 'Hmcл' THEN '2'
    WHEN 'M в' THEN 'M в'
    WHEN 'M сл' THEN '4'
    WHEN 'M ср' THEN '5'
    WHEN 'Mсл' THEN '4'
    WHEN 'М сл' THEN '4'
    WHEN 'М ср' THEN '5'
    WHEN 'Мв' THEN 'Мв'
    WHEN 'Мсл' THEN '4'
    ELSE refraction
END;