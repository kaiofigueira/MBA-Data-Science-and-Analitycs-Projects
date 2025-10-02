create database livraria;
use livraria;

create table autores (
	id int primary key auto_increment,
	nome varchar(200)
);

create table generos (
	id int primary key auto_increment,
	nome varchar(200)
);

create table livros (
	id int primary key auto_increment,
	nome varchar(200),
    idioma varchar(100),
    ano_publicacao int,
    vendas decimal(10,2),
    autor_id int,
    genero_id int,
    foreign key (autor_id) references autores(id),
    foreign key (genero_id) references generos(id)
);

create table comentarios (
    id int primary key auto_increment,
    livro_id int,
    nome varchar(100),
    sobrenome varchar(100),
    comentario text,
    foreign key (livro_id) references livros(id)
);

select * from autores;
select * from generos;
select * from livros order by id;
select * from comentarios;
SELECT * FROM comentarios LIMIT 5000;

drop table comentarios;

-- buscar todos os livros publicados após o ano 2010 
select nome from livros
where ano_publicacao > 2010;

-- listar livros com vendas acima de 100
select nome, vendas
from livros
where vendas > 100; 

-- listar livros com nome do autor
select l.nome as livro, a.nome as autor
from livros l
join autores a on l.autor_id = a.id;

-- contar quantos livros cada autor publicou 
select autor_id, count(*) as total_livros
from livros
group by autor_id;

-- listar livro, autor e gênero
select l.nome as livro, a.nome as autor, g.nome as genero
from livros l
join autores a on l.autor_id = a.id
join generos g on l.genero_id = g.id;

-- livros cujo gênero seja exatamente 'desconhecido'
select l.*
from livros l
join generos g on l.genero_id = g.id
where g.nome = 'desconhecido';

-- resumo estatístico
select
  min(vendas) as min_vendas,
  max(vendas) as max_vendas,
  avg(vendas) as media_vendas
from livros
where vendas is not null;

-- vendas por autor
select 
    a.nome as autor, 
    sum(l.vendas) as total_vendas, 
    count(l.id) as livros_publicados
from livros l
join autores a on l.autor_id = a.id
group by a.nome
order by total_vendas desc;

-- Listar top 5 gêneros com maior faturamento total
select 
    g.nome as genero,
    sum(l.vendas) as faturamento_total
from livros l
join generos g on l.genero_id = g.id
group by g.nome
order by faturamento_total desc
limit 5;

-- total de vendas por autor e por gênero
select 
  a.nome as autor,
  g.nome as genero,
  count(l.id) as total_livros,
  sum(l.vendas) as total_vendas,
  avg(l.vendas) as media_vendas
from livros l
join autores a on l.autor_id = a.id
join generos g on l.genero_id = g.id
where l.vendas is not null
group by a.nome, g.nome
order by total_vendas desc;
